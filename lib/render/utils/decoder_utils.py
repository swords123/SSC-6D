import torch
import json
import os, sys
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import numpy as np

def load_decoder(experiment_directory, checkpoint_num=None, parallel=True):
    specs_filename = os.path.join(experiment_directory, "specs.json")

    if not os.path.isfile(specs_filename):
        raise Exception(
            'The experiment directory does not include specifications file "specs.json"'
        )

    specs = json.load(open(specs_filename))

    arch = __import__("." + specs["NetworkArch"], fromlist=["Decoder"])

    latent_size = specs["CodeLength"]
    decoder = arch.Decoder(latent_size, **specs["NetworkSpecs"]).cuda()

    if parallel:
       decoder = torch.nn.DataParallel(decoder)

    if checkpoint_num != None:
        saved_model_state = torch.load(
            os.path.join(experiment_directory, "ModelParameters", checkpoint_num + ".pth")
        )

        saved_model_epoch = saved_model_state["epoch"]

        if not parallel:
            state_dict = saved_model_state["model_state_dict"]
            new_state_dict = dict()
            for key,value in state_dict.items():
                new_state_dict[key.replace('module.', '')] = value
            decoder.load_state_dict(new_state_dict)
        else:
            decoder.load_state_dict(saved_model_state["model_state_dict"])
    return decoder

def decode_sdf(decoder, latent_vector, points, clamp_dist=0.1, MAX_POINTS=100000, no_grad=False):
    start, num_all = 0, points.shape[0]
    output_list = []
    while True:
        end = min(start + MAX_POINTS, num_all)
        if latent_vector is None:
            inputs = points[start:end]
        else:
            latent_repeat = latent_vector.expand(end - start, -1)
            inputs = torch.cat([latent_repeat, points[start:end]], 1)
        #sdf_batch = decoder.inference(inputs)
        sdf_batch = decoder(inputs)
        start = end
        if no_grad:
            sdf_batch = sdf_batch.detach()
        output_list.append(sdf_batch)
        if end == num_all:
            break
    sdf = torch.cat(output_list, 0)

    if clamp_dist != None:
        sdf = torch.clamp(sdf, -clamp_dist, clamp_dist)
    return sdf

def decode_sdf_gradient(decoder, latent_vector, points, clamp_dist=0.1, MAX_POINTS=100000, no_grad=False):
    start, num_all = 0, points.shape[0]
    output_list = []
    while True:
        end = min(start + MAX_POINTS, num_all)
        points_batch = points[start:end]
        sdf = decode_sdf(decoder, latent_vector, points_batch, clamp_dist=clamp_dist)
        start = end
        grad_tensor = torch.autograd.grad(outputs=sdf, inputs=points_batch, grad_outputs=torch.ones_like(points_batch), create_graph=True, retain_graph=True)
        grad_tensor = grad_tensor[0]
        if no_grad:
            grad_tensor = grad_tensor.detach()
        output_list.append(grad_tensor)
        if end == num_all:
            break
    grad_tensor = torch.cat(output_list, 0)
    return grad_tensor


