import torch
import numpy as np

MIN_LOG_STD = np.log(1e-3)
"""
MSE loss between prediction and target, no logstdariance

input: 
	pred: Nx3 vector of network displacement output
	targ: Nx3 vector of gt displacement
output:
	loss: Nx3 vector of MSE loss on x,y,z
"""


def loss_mse(pred, targ):
    loss = (pred - targ).pow(2)
    return loss


"""
Log Likelihood loss, with logstdariance (only support diag logstd)

input:
	pred: Nx3 vector of network displacement output
	targ: Nx3 vector of gt displacement
	pred_logstd: Nx3 vector of log(sigma) on the diagonal entries
output:
	loss: Nx3 vector of likelihood loss on x,y,z

resulting pred_logstd meaning:
pred_logstd:(Nx3) u = [log(sigma_x) log(sigma_y) log(sigma_z)]
"""


def loss_distribution_diag(pred, pred_logstd, targ):
    pred_logstd = torch.maximum(pred_logstd,
                                MIN_LOG_STD * torch.ones_like(pred_logstd))
    # 判断exp运算是否越界
    # 若越界定位数据位置
    if torch.any(torch.isinf(torch.exp(2 * pred_logstd))):
        # inf -> mean
        # return loss_mse(pred, targ)
        pred_logstd[torch.isinf(torch.exp(2 * pred_logstd))] = torch.mean(
            pred_logstd[~torch.isinf(torch.exp(2 * pred_logstd))])
    loss = (
        (pred - targ).pow(2)) / (2 * torch.exp(2 * pred_logstd)) + pred_logstd

    return loss


"""
Log Likelihood loss, with logstdariance (support full logstd)
(NOTE: output is Nx1)

input:
	pred: Nx3 vector of network displacement output
	targ: Nx3 vector of gt displacement
	pred_logstd: Nxk logstdariance parametrization
output:
	loss: Nx1 vector of likelihood loss

resulting pred_logstd meaning:
DiagonalParam:
pred_logstd:(Nx3) u = [log(sigma_x) log(sigma_y) log(sigma_z)]
PearsonParam:
pred_logstd (Nx6): u = [log(sigma_x) log(sigma_y) log(sigma_z)
										 rho_xy, rho_xz, rho_yz] (Pearson correlation coeff)
FunStuff
"""



"""
Select loss function based on epochs
all variables on gpu
output:
	loss: Nx3
"""


def get_loss(pred, pred_logstd, targ, epoch, switch_epoch=10):
    """
		if epoch < 10:
				loss = loss_mse(pred, targ)
		else:
				loss = loss_distribution_diag(pred, pred_logstd, targ)
		"""

    # if epoch < switch_epoch:

    pred_logstd = pred_logstd.detach()

    loss = loss_distribution_diag(pred, pred_logstd, targ)

    # loss = loss_mse(pred, targ)
    return loss

class Loss(object):
    def __call__(self, pred, pred_logstd, targ, epoch):
        return get_loss(pred, pred_logstd, targ, epoch)