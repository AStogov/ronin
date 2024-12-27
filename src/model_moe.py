import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from model_resmlp import TwoLayerModel


class MoeModel(nn.Module):

    def __init__(self,
                 model_para,
                 num_experts=4,
                 input_dim=200,
                 topk=2,
                 capacity_factor=2.4):
        super(MoeModel, self).__init__()
        self.expert_proportions = None
        self.topk = topk
        self.input_channel = model_para['input_channel']
        self.capacity_factor = capacity_factor
        self.num_experts = num_experts
        self.expert = nn.ModuleList(
            [TwoLayerModel(model_para) for _ in range(num_experts)])
        self.gate = nn.Sequential(nn.Linear(input_dim, num_experts),
                                  nn.Softmax(dim=-1))
        self._initialize()

    def _initialize(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight,
                                        mode="fan_out",
                                        nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def get_num_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_modules(self):
        for idx, m in enumerate(self.modules()):
            print(idx, '->', m)

    def get_gate(self, x):
        gate = self.gate(x)
        gate = torch.einsum('ijk->ik', gate) / self.input_channel
        return gate

    def get_experts(self, x):
        # experts: [batch_size, num_experts, output_dim]
        # wrote expert(x)[1], but made no difference
        experts = torch.stack([expert(x)[0] for expert in self.expert], dim=1)
        return experts

    def get_gate_with_capacity(self, gate):
        topk_values, topk_indices = torch.topk(gate, self.topk, dim=-1)
        capacity = self.capacity_factor * gate.size(0) / self.num_experts
        tophot = torch.zeros_like(gate).scatter(1, topk_indices, 1)
        topsum = torch.cumsum(tophot, dim=0)
        mask = topsum > capacity
        gate = torch.where(mask, torch.zeros_like(gate), gate)
        topk_values, topk_indices = torch.topk(gate, self.topk, dim=-1)
        gate = torch.zeros_like(gate).scatter(1, topk_indices, topk_values)
        return gate

    def forward(self, x):
        # x: (batch_size, input_channel, input_dim)
        gate = self.get_gate(x)
        # keep the topk experts

        # expert: [batch_size, num_experts, output_dim]
        experts = self.get_experts(x)

        # gate: [batch_size, num_experts]
        gate = self.get_gate_with_capacity(gate)

        with torch.no_grad():
            self.expert_proportions = torch.mean(gate, dim=0)

        # output = expert * gate
        output = torch.einsum('ijk,ij->ik', experts, gate)

        # Calculate covariance matrix for experts
        mean_experts = torch.mean(experts, dim=1, keepdim=True)
        experts_centered = experts - mean_experts  # Center the data
        covariance = torch.einsum('ijk,ijl->ikl', experts_centered, experts_centered) / (experts.size(1) - 1)

        # Extract the diagonal (variance) from the covariance matrix
        diag_covariance = torch.diagonal(covariance, dim1=-2, dim2=-1)

        return output, diag_covariance



if __name__ == '__main__':
    model_para = {
        "input_len": 100,
        "input_channel": 6,
        "patch_len": 25,
        "feature_dim": 128,
        "out_dim": 3,
        "active_func": "GELU",
        "extractor": { # include: Feature Convert & ResMLP Module in the paper Fig. 3.
            "name": "ResMLP",
            "layer_num": 5,
            "expansion": 4,
            "dropout": 0.2,
        },
        "reg": { # Regression in the paper Fig.3
            "name": "MeanMLP",
            # "name": "MaxMLP",
            "layer_num": 3,
        }
    }
    device = torch.device('mps')
    net = MoeModel(model_para).to(device)  # initialize the model
    x = torch.rand([512, 6,
                    100]).to(device)  # batch_size, input_channel, input_len
    y, var = net(x)  # output: [batch_size, 3]
    y_ = torch.randn([512, 3]).to(device)
    # print every expert's mse compare to y_
    experts = net.get_experts(x)
    print(experts)
    with torch.no_grad():
        for i in range(4):
            print(F.mse_loss(experts[:, i], y_))

    print(net.expert_proportions)
