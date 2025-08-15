from util import prepare_channel_tx_ris, find_indices, discretize_phase, compute_noma_performance
from core import RISnetNOMA, RISnetPartialCSINOMA, RTChannelsNOMA
import torch.optim as optim
import torch
from torch.utils.data import DataLoader
import numpy as np
from params import params4noma as params
import argparse
import datetime
from pathlib import Path
from torch.optim.lr_scheduler import ReduceLROnPlateau

tb = True
try:
    from tensorboardX import SummaryWriter
except:
    tb = False
record = False and tb


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--tsnr")
    parser.add_argument("--ris_shape")
    parser.add_argument("--weights")
    parser.add_argument("--lr")
    parser.add_argument("--record")
    parser.add_argument("--device")
    parser.add_argument("--partialcsi")
    parser.add_argument("--trainingchannelpath")
    parser.add_argument("--testingchannelpath")
    parser.add_argument("--name")
    args = parser.parse_args()
    if args.tsnr is not None:
        params["tsnr"] = float(args.tsnr)
    if args.lr is not None:
        params["lr"] = float(args.lr)
    if args.weights is not None:
        weights = args.weights.split(',')
        params["alphas"] = np.array([float(w) for w in weights])
    if args.ris_shape is not None:
        ris_shape = args.ris_shape.split(',')
        params["ris_shape"] = tuple([int(s) for s in ris_shape])
    if args.record is not None:
        record = args.record == "True"
    if args.partialcsi is not None:
        params["partial_csi"] = args.partialcsi == "True"
    if args.device is not None:
        device = args.device
    else:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if args.trainingchannelpath is not None:
        params['channel_ris_rx_path'] = args.trainingchannelpath
    if args.testingchannelpath is not None:
        params['channel_ris_rx_testing_path'] = args.testingchannelpath
    tb = tb and record

    if record:
        now = datetime.datetime.now()
        if args.name is not None:
            dt_string = args.name
        else:
            dt_string = now.strftime("%d-%m-%Y_%H-%M-%S")
        Path(params["results_path"] + dt_string).mkdir(parents=True, exist_ok=True)
        params["results_path"] = params["results_path"] + dt_string + "/"

    params["discrete_phases"] = params["discrete_phases"].to(device)

    if params["partial_csi"]:
        model = RISnetPartialCSINOMA(params).to(device)
    else:
        model = RISnetNOMA(params).to(device)
    # model = torch.compile(model)

    channel_tx_ris, channel_tx_ris_pinv = prepare_channel_tx_ris(params, device)
    training_set = RTChannelsNOMA(params, channel_tx_ris_pinv, device)
    test_set = RTChannelsNOMA(params, channel_tx_ris_pinv, device, test=True)
    result_name = "ris_" + str(params['tsnr']) + "_" + str(params['ris_shape']) + '_' + str(params['alphas']) + "_"
    training_loader = DataLoader(dataset=training_set, batch_size=params['batch_size'], shuffle=True)
    test_batch_size = test_set.group_definition.shape[0]
    test_loader = DataLoader(dataset=test_set, batch_size=test_batch_size, shuffle=True)
    losses = list()
    if tb and record:
        writer = SummaryWriter(logdir=params["results_path"])
    counter = 1
    optimizer = optim.Adam(model.parameters(), params['lr'])
    scheduler = ReduceLROnPlateau(optimizer, 'min', min_lr=1e-8, patience=100, factor=0.5)

    if params["partial_csi"]:
        sensor_indices = find_indices(params["indices_sensors"], params["indices_sensors"], params["ris_shape"][1])
    else:
        sensor_indices = np.arange(params["ris_shape"][0] * params["ris_shape"][1])

    if params["mutual_coupling"]:
        sii = torch.load("data/mutual_coupling.pt")
    else:
        sii = None

    # Training with WMMSE precoder
    training_underway = True
    while training_underway:
        for batch in training_loader:
            model.train()
            sample_indices, model_input, data_rate, channels_ris_rx, channels_direct, location = batch

            optimizer.zero_grad()

            nn_raw_output = model(model_input[:, :, sensor_indices])
            constraint, qd_ratio, power = compute_noma_performance(nn_raw_output, channel_tx_ris, channels_ris_rx,
                                                                   channels_direct, data_rate, sii, params, device)

            loss = torch.mean(power + 100 * constraint)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
            for name, param in model.named_parameters():
                if torch.isnan(param.grad).any():
                    print("nan gradient found")
            optimizer.step()
            scheduler.step(loss)

            with torch.no_grad():
                model.eval()
                for batch in test_loader:
                    (sample_indices, model_input_test, data_rate_test,
                     channels_ris_rx_test, channels_direct_test, location_test) = batch
                    nn_raw_output_test = model(model_input_test[:, :, sensor_indices])
                    _, qd_ratio_test, power_test = compute_noma_performance(nn_raw_output_test, channel_tx_ris,
                                                                            channels_ris_rx_test,
                                                                            channels_direct_test, data_rate_test,
                                                                            sii, params, device)

                    discrete_phases = discretize_phase(nn_raw_output_test, np.pi, device)
                    _, qd_ratio_test_pi, power_test_pi = compute_noma_performance(discrete_phases, channel_tx_ris,
                                                                                  channels_ris_rx_test,
                                                                                  channels_direct_test, data_rate_test,
                                                                                  sii, params, device)

                    discrete_phases = discretize_phase(nn_raw_output_test, np.pi / 2, device)
                    _, qd_ratio_test_pid2, power_test_pid2 = compute_noma_performance(discrete_phases, channel_tx_ris,
                                                                                      channels_ris_rx_test,
                                                                                      channels_direct_test, data_rate_test,
                                                                                      sii, params, device)

            print('Epoch {epoch}, '
                  'transmit power = {loss}, QD ratio = {qd}'.format(loss=power.mean(),
                                                   epoch=counter, qd=qd_ratio))

            if tb and record:
                writer.add_scalar("Training/power", power.item(), counter)
                writer.add_scalar("Training/ratio", qd_ratio.item(), counter)
                writer.add_scalar("Training/learning_rate",
                                  optimizer.param_groups[0]["lr"], counter)
                writer.add_scalar("Testing/power_continuous", power_test.mean().item(), counter)
                writer.add_scalar("Testing/power_pi", power_test_pi.mean().item(), counter)
                writer.add_scalar("Testing/power_pid2", power_test_pid2.mean().item(), counter)
                writer.add_scalar("Testing/ratio_continuous", qd_ratio_test.item(), counter)
                writer.add_scalar("Testing/ratio_pi", qd_ratio_test_pi.item(), counter)
                writer.add_scalar("Testing/ratio_pid2", qd_ratio_test_pid2.item(), counter)

            counter += 1
            if counter > params["iter_wmmse"]:
                training_underway = False

            if record and counter % params['wmmse_saving_frequency'] == 0:
                torch.save(model.state_dict(), params['results_path'] + result_name +
                           '{iter}'.format(iter=counter))
                if tb:
                    writer.flush()
