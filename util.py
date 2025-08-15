import numpy as np
import torch
from scipy.linalg import eigh
import copy
from joblib import Parallel, delayed
from torch import nn
from torch import linalg as LA

solver = True
try:
    from scipy.optimize import fsolve
except:
    solver = False


def mmse_precoding(channel, params, device='cpu'):
    if type(channel) is np.ndarray:
        channel = torch.from_numpy(channel).to(device)
    eye = torch.eye(channel.shape[1]).repeat((channel.shape[0], 1, 1)).to(device)
    p = channel.transpose(1, 2).conj() @ torch.linalg.inv(channel @ channel.transpose(1, 2).conj() +
                                                          1 / params['tsnr'] * eye)
    trace = torch.sum(torch.diagonal((p @ p.transpose(1, 2).conj()), dim1=1, dim2=2).real, dim=1, keepdim=True)
    p = p / torch.unsqueeze(torch.sqrt(trace), dim=2)
    return p


def cp2array_risnet(cp, factor=1, mean=0, device="cpu"):
    # Input: (batch, antenna)
    # Output: (batch, feature, antenna))
    array = torch.cat([(cp.abs() - mean) * factor, cp.angle() * 0.55], dim=1)

    return array.to(device)


def prepare_channel_direct_features(channel_direct, channel_tx_ris_pinv, params, device='cpu'):
    equivalent_los_channel = channel_direct @ channel_tx_ris_pinv
    return cp2array_risnet(equivalent_los_channel, 1 / params['std_direct'], params["mean_direct"], device)


def weighted_sum_rate(complete_channel, precoding, weights, params):
    channel_precoding = complete_channel @ precoding
    channel_precoding = torch.square(channel_precoding.abs())
    wsr = 0
    num_users = channel_precoding.shape[1]
    for user_idx in range(num_users):
        wsr += weights[:, user_idx] * torch.log2(1 + channel_precoding[:, user_idx, user_idx] /
                                                       (torch.sum(channel_precoding[:, user_idx, :], dim=1)
                                                        - channel_precoding[:, user_idx, user_idx]
                                                        + 1 / params["tsnr"]))
    return wsr


def test_model(complete_channel, precoding, params):
    if type(complete_channel) is np.ndarray:
        complete_channel = torch.from_numpy(complete_channel).cfloat()

    if type(precoding) is np.ndarray:
        precoding = torch.from_numpy(precoding).cfloat()

    channel_precoding = complete_channel @ precoding
    channel_precoding = torch.square(channel_precoding.abs())
    data_rates = list()
    num_users = channel_precoding.shape[1]
    for user_idx in range(num_users):
        data_rates.append(torch.log2(1 + channel_precoding[:, user_idx, user_idx] /
                                     (torch.sum(channel_precoding[:, user_idx, :], dim=1)
                                      - channel_precoding[:, user_idx, user_idx]
                                      + 1 / params["tsnr"])).cpu().detach().numpy())
    return channel_precoding.cpu().detach().numpy(), data_rates


def array2phase_shifts(phase_shifts):
    # Input: (batch, 1, width, height)
    # Output: (batch, antenna, antenna)
    p = torch.flatten(phase_shifts[:, 0, :, :], start_dim=1, end_dim=2)
    p = torch.diag_embed(torch.exp(1j * p))
    return p


def compute_wmmse_v_v2(h_as_array, init_v, tx_power, noise_power, params, num_iters=500):
    num_users, num_tx_antennas = h_as_array.shape
    h_list = [h_as_array[user_idx: (user_idx + 1), :] for user_idx in range(num_users)]
    v_list = [init_v[:, user_idx: (user_idx + 1)] for user_idx in range(num_users)]
    w_list = [1 for _ in range(num_users)]
    for iter in range(num_iters):
        w_list_old = copy.deepcopy(w_list)

        # Step 2
        u_list = list()
        for user_idx in range(num_users):
            inv_hvvhi = (1 / (np.sum([np.real(h_list[user_idx] @ v
                                              @ v.transpose().conj() @ h_list[user_idx].transpose().conj())
                                      for v in v_list]) + noise_power))
            u_list.append(inv_hvvhi * h_list[user_idx] @ v_list[user_idx])

        # Step 3
        for user_idx in range(num_users):
            w_list[user_idx] = 1 / np.real(1 - u_list[user_idx].transpose().conj()
                                           @ h_list[user_idx] @ v_list[user_idx])

        # Step 4
        mmu = sum([alpha * h.transpose().conj() @ u @ w @ u.transpose().conj() @ h for alpha, h, u, w, in
                   zip(params["alphas"], h_list, u_list, w_list)])
        mphi = sum([alpha ** 2 * h.transpose().conj() @ u @ w ** 2 @ u.transpose().conj() @ h for alpha, h, u, w in
                    zip(params["alphas"], h_list, u_list, w_list)])

        try:
            lambbda, d = eigh(mmu)
        except:
            break
        lambbda = np.real(lambbda)
        phi = d.transpose().conj() @ mphi @ d
        phi = np.real(np.diag(phi))
        if solver:
            mu = fsolve(solve_mu, 0, args=(phi, lambbda, tx_power))
        else:
            raise ImportError('scipy.optimize.fsolve cannot be imported.')
        mv = np.linalg.inv(mmu + mu * np.eye(num_tx_antennas))

        v_list = [alpha * mv @ h.transpose().conj() @ u @ w for alpha, h, u, w in
                  zip(params["alphas"], h_list, u_list, w_list)]

        if np.sum([np.abs(w - w_old) for w, w_old in zip(w_list, w_list_old)]) < np.abs(w_list[0]) / 100 and iter > 500:
            break

    precoding = np.hstack(v_list)
    power = np.sum(np.abs(precoding) ** 2)
    return precoding / np.sqrt(power)


def wmmse_precoding(h, tx_power, noise_power, num_tx_antennas, params, num_cpus=1):
    num_samples = h.shape[0]
    res = Parallel(n_jobs=num_cpus)(delayed(compute_wmmse_v_v2)(h[sample_id, :, :], tx_power, noise_power,
                                                                params)
                                    for sample_id in range(num_samples))
    v = np.stack(res, axis=0)
    return v


def solve_mu(mu, *args):
    phi = args[0]
    lambbda = args[1]
    p = args[2]
    return np.sum(phi / (lambbda + mu + 1e-3) ** 2) - p


def compute_complete_channel(channel_tx_ris, nn_output, channel_ris_rx, channel_direct):
    phi = torch.exp(1j * nn_output)
    complete_channel = (channel_ris_rx * phi) @ channel_tx_ris + channel_direct
    return complete_channel


def compute_complete_channel_mutual(channel_tx_ris, nn_output, channel_ris_rx, channel_direct, sii, params, device="cpu"):
    phi = torch.exp(1j * nn_output)
    identity = torch.eye(params["num_ris_antennas"]).to(device)
    to_be_inversed = identity - phi[:, :, None] * sii
    x = torch.linalg.solve(to_be_inversed[:, 0, :, :], channel_ris_rx, left=False)
    complete_channel = x * phi @ channel_tx_ris + channel_direct
    return complete_channel


def prepare_channel_tx_ris(params, device="cpu"):
    channel_tx_ris = torch.load(params['channel_tx_ris_path'], map_location=torch.device(device)).cfloat()
    channel_tx_ris = channel_tx_ris[:params["ris_shape"][0] * params["ris_shape"][1], :]
    channel_tx_ris_pinv = torch.linalg.pinv(channel_tx_ris)
    return channel_tx_ris, channel_tx_ris_pinv


def find_indices(x, y, n_cols):
    x, y = np.meshgrid(x, y)
    return (x + y * n_cols).flatten()


def discretize_phase(phases, granularity, device="cpu"):
    eligible_phases = torch.arange(0, 2 * torch.pi, granularity).to(device)
    eligible_vectors = torch.exp(1j * eligible_phases)
    eligible_vectors = eligible_vectors[None, :, None]
    phis = torch.exp(1j * phases)
    prod = eligible_vectors.real * phis.real + eligible_vectors.imag * phis.imag
    chosen_phases = torch.argmax(prod, dim=1, keepdim=True)
    chosen_phases = chosen_phases.float()
    chosen_phases *= granularity
    return chosen_phases


def average_precoding_power(precoding):
    precoding_norm = torch.norm(precoding, dim=1)
    power = torch.square(precoding_norm)
    avg_power = torch.mean(torch.sum(power, dim=1))
    # print(f"Average precoding power: {avg_power.item()}")
    return avg_power


def quasi_degradation_penalty(complete_channel, data_rates):
    h1 = complete_channel[:, :1, :].transpose(dim0=1, dim1=2)
    h2 = complete_channel[:, 1:, :].transpose(dim0=1, dim1=2)
    h1h = torch.adjoint(h1)
    h2h = torch.adjoint(h2)
    cos2psi = (torch.squeeze(h1h @ h2 @ h2h @ h1).real /
               torch.squeeze(torch.linalg.norm(h1, dim=1) ** 2 * torch.linalg.norm(h2, dim=1) ** 2))
    mask_small = torch.greater(torch.tensor(0.00001), cos2psi)
    cos2psi[mask_small] = torch.tensor(0.00001).type(torch.DoubleTensor)
    mask_big = torch.greater(cos2psi, torch.tensor(0.99999))
    cos2psi[mask_big] = torch.tensor(0.99999).type(torch.DoubleTensor)
    r1 = 2 ** (data_rates[:, 0] - 1)
    r2 = 2 ** (data_rates[:, 1] - 1)
    q = (1 + r1) / cos2psi - (r1 * cos2psi) / (1 + r2 * (1 - cos2psi)) ** 2
    diff = q - torch.squeeze(torch.linalg.norm(h1, dim=1) ** 2 /
                             torch.linalg.norm(h2, dim=1) ** 2)
    return torch.nn.functional.relu(diff)


def precoding_qd(complete_channel, data_rates, params):
    sigma = np.sqrt(1 / params["tsnr"])
    h1 = complete_channel[:, :1, :].transpose(dim0=1, dim1=2).conj()
    h2 = complete_channel[:, 1:, :].transpose(dim0=1, dim1=2).conj()
    h1h = torch.adjoint(h1)
    h2h = torch.adjoint(h2)
    cos2psi = (torch.squeeze(h1h @ h2 @ h2h @ h1).real /
               torch.squeeze(torch.linalg.norm(h1, dim=1) ** 2 * torch.linalg.norm(h2, dim=1) ** 2))
    r1 = 2 ** (data_rates[:, 0] - 1)
    r2 = 2 ** (data_rates[:, 1] - 1)
    e1 = h1 / torch.linalg.norm(h1, dim=1)[:, :, None]
    e2 = h2 / torch.linalg.norm(h2, dim=1)[:, :, None]
    alpha1 = torch.sqrt(r1 * sigma ** 2 / torch.squeeze(torch.norm(h1, dim=1)) ** 2 / (1 + r2 * (1 - cos2psi)) ** 2)
    alpha2 = torch.sqrt(r2 * sigma ** 2 / torch.squeeze(torch.norm(h2, dim=1)) ** 2 +
                        r1 * sigma ** 2 / torch.squeeze(torch.norm(h1, dim=1)) ** 2 *
                        r2 * cos2psi / (1 + r2 * (1 - cos2psi)) ** 2)
    w1 = alpha1[:, None, None] * ((1 + r2)[:, None, None] * e1 - r2[:, None, None] * e2.transpose(dim0=1, dim1=2).conj() @ e1 * e2)
    w2 = alpha2[:, None, None] * e2
    return torch.cat([w1.conj(), w2.conj()], dim=2)


def calc_sharp_penalty(penalty):
    return torch.relu((penalty / 5) ** 0.3 + penalty)


def compute_noma_performance(nn_raw_output, channel_tx_ris, channels_ris_rx, channels_direct, data_rate, sii, params,
                             device="cpu"):
    if sii is None:
        complete_channel = compute_complete_channel(channel_tx_ris, nn_raw_output,
                                                    channels_ris_rx, channels_direct)
    else:
        complete_channel = compute_complete_channel_mutual(channel_tx_ris, nn_raw_output,
                                                           channels_ris_rx, channels_direct, sii, params, device)
    constraint = calc_sharp_penalty(quasi_degradation_penalty(complete_channel, data_rate))
    qd_ratio = torch.sum(constraint == 0) / constraint.shape[0]
    precoding = precoding_qd(complete_channel, data_rate, params)
    power = average_precoding_power(precoding)
    return constraint, qd_ratio, power


def compute_sdma_performance(nn_raw_output, channel_tx_ris, channels_ris_rx, channels_direct, precoding, sii, weights,
                             params, device="cpu"):
    if sii is None:
        complete_channel = compute_complete_channel(channel_tx_ris, nn_raw_output,
                                                    channels_ris_rx, channels_direct)
    else:
        complete_channel = compute_complete_channel_mutual(channel_tx_ris, nn_raw_output,
                                                           channels_ris_rx, channels_direct, sii, params, device)
    wsr = weighted_sum_rate(complete_channel, precoding, weights, params)
    return wsr
