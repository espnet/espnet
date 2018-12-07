import torch
import pickle
from e2e_tts_th import Tacotron2, Tacotron2Loss
from tts_pytorch import CustomConverter

import logging


def TacotronRewardLoss(tts_model_file, idim=None, odim=None, train_args=None,
                       use_masking=False, bce_pos_weight=20.0,
                       spk_embed_dim=None):

    # define output activation function
    if hasattr(train_args, 'output_activation'):
        if train_args.output_activation is None:
            output_activation_fn = None
        elif hasattr(torch.nn.functional, train_args.output_activation):
            output_activation_fn = getattr(
                torch.nn.functional, train_args.output_activation
            )
        else:
            raise ValueError(
                'there is no such an activation function. (%s)' %
                train_args.output_activation
            )
    else:
        output_activation_fn = None

    # TACOTRON CYCLE-CONSISTENT LOSS HERE
    # Define model
    tacotron2 = Tacotron2(
        idim=idim,
        odim=odim,
        spk_embed_dim=spk_embed_dim,
        embed_dim=train_args.embed_dim,
        elayers=train_args.elayers,
        eunits=train_args.eunits,
        econv_layers=train_args.econv_layers,
        econv_chans=train_args.econv_chans,
        econv_filts=train_args.econv_filts,
        dlayers=train_args.dlayers,
        dunits=train_args.dunits,
        prenet_layers=train_args.prenet_layers,
        prenet_units=train_args.prenet_units,
        postnet_layers=train_args.postnet_layers,
        postnet_chans=train_args.postnet_chans,
        postnet_filts=train_args.postnet_filts,
        adim=train_args.adim,
        aconv_chans=train_args.aconv_chans,
        aconv_filts=train_args.aconv_filts,
        output_activation_fn=output_activation_fn,
        cumulate_att_w=train_args.cumulate_att_w,
        use_batch_norm=train_args.use_batch_norm,
        use_concate=train_args.use_concate,
        dropout=train_args.dropout_rate,
        zoneout=train_args.zoneout_rate,
    )
    # load trained model parameters
    logging.info('reading model parameters from ' + tts_model_file)
    tacotron2.load_state_dict(
        torch.load(tts_model_file, map_location=lambda storage, loc: storage))
    # Set to eval mode
    tacotron2.eval()
    # Define loss
    loss = Tacotron2Loss(
        model=tacotron2,
        use_masking=use_masking,
        bce_pos_weight=bce_pos_weight,
        report=False,
        # These two are needed together
        reduce_loss=False,
        use_bce_loss=False
    )

    loss.eval()
    loss.train_args = train_args
    return loss


def load_tacotron_loss(tts_model_conf, tts_model_file):

    # Read model
    with open(tts_model_conf, 'rb') as f:
        idim_taco, odim_taco, train_args_taco = pickle.load(f)
    # Load loss
    return TacotronRewardLoss(
        tts_model_file,
        idim=idim_taco,
        odim=odim_taco,
        train_args=train_args_taco,
    )


def sanity_check_json(valid_json):

    # Sanity check for first sample
    sample = valid_json.values()[0]
    assert len(sample['input']) == 3, \
        "Expected three inputs in data asr-mel tts-mel and x-vector"
    assert (
        sample['input'][0]['shape'][1] ==
        sample['input'][1]['shape'][1]
    ), "Expected inputs 0 and 1 (asr-mel, tts-mel) to be same size"


def convert_espnet_to_taco_batch(x, ys, batch_size, n_samples_per_input,
                                 ngpu, use_speaker_embedding=False):
    """
    Convert data to format suitable for Tacotron, borrow code from
    needs xs, ilens, ys, spembs for tacotron loss
    src/tts/tts_pytorch.py:CustomConverter:__call__
    """

    assert use_speaker_embedding, \
        "use_speaker_embedding=False not supported yet"

    # Number of gpus
    if ngpu == 1:
        gpu_id = range(ngpu)
    elif ngpu > 1:
        gpu_id = range(ngpu)
    else:
        gpu_id = [-1]

    # Tacotron converter
    taco_converter = CustomConverter(
        gpu_id,
        use_speaker_embedding=use_speaker_embedding
    )

    # Reformat batch
    samples_batch = []
    for sample_index in range(n_samples_per_input):
        batch_sample = []
        for batch_index in range(batch_size):
            text_sample = ys[batch_index + sample_index]
            content = {
                u'input': [
                    {
                        u'feat': x[batch_index][1]['input'][1]['feat'],
                        u'name': u'input1',
                        u'shape': x[batch_index][1]['input'][1]['shape']
                    },
                    {
                        u'feat': x[batch_index][1]['input'][2]['feat'],
                        u'name': u'input2',
                        u'shape': x[batch_index][1]['input'][2]['shape']
                    }
                ],
                u'output': [{
                    u'name': u'target1',
                    u'shape': [
                        len(text_sample),
                        x[batch_index][1]['output'][0][u'shape'][1]
                    ],
                    u'text': None,
                    u'token': None,
                    u'tokenid': " ".join(
                        map(str, list(text_sample.data.cpu().numpy()))
                    )
                }],
                u'utt2spk': x[batch_index][1][u'utt2spk']
            }
            batch_sample.append((x[batch_index][0], content))
        samples_batch.append(taco_converter([batch_sample], True))
    return samples_batch


def extract_tacotron_features(x, ys, n_samples_per_input, num_gpu): 

    # Expand the batch for each sample, extract tacotron 
    expanded_x = []
    import copy
    for example_index, example_x in enumerate(x):
        for n in range(n_samples_per_input):
    
            # Assumes samples are placed consecutively
            text_sample = ys[example_index*n_samples_per_input + n]
            new_example_x = copy.deepcopy(example_x)
    
            # Remove ASR features
            del new_example_x[1]['input'][0]
    
            # Replace output sequence
            new_example_x[1]['output'][0]['shape'][0] = len(text_sample)
            new_example_x[1]['output'][0]['text'] = None
            new_example_x[1]['output'][0]['token'] = None
            new_example_x[1]['output'][0]['tokenid'] = " ".join(
                map(str, list(text_sample.data.cpu().numpy()))
            )
            expanded_x.append(new_example_x)
    
    # Number of gpus
    if num_gpu == 1:
        gpu_id = range(num_gpu)
    elif num_gpu > 1:
        gpu_id = range(num_gpu)
    else:
        gpu_id = [-1]
    
    # Construct a Tacotron batch from ESPNet batch and the samples
    # Tacotron converter
    from tts_pytorch import CustomConverter
    taco_converter = CustomConverter(
        gpu_id,
        use_speaker_embedding=True
    )
    return taco_converter([expanded_x])
