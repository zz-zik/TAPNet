from .TAPNet import TAPNet
from .losses import SetCriterion_Ahead
from .matcher import build_matcher_crowd
from .losses import SetCriterion_Crowd


def build_model(cfg, training=False):
    model = TAPNet(cfg)
    if not training:
        return model
    # weight_dict = {'loss_ce': 1, 'loss_points': cfg.point_loss_coef, 'loss_aux': cfg.loss_aux}
    weight_dict = {'loss_ce': 1, 'loss_points': cfg.point_loss_coef}
    matcher = build_matcher_crowd(cfg)
    # 创建criterion损失函数
    criterion_crowd = SetCriterion_Crowd(num_classes=1, matcher=matcher, weight_dict=weight_dict,
                                         eos_coef=cfg.eos_coef,
                                         aux_kwargs={'AUX_NUMBER': cfg.aux_number,
                                                     'AUX_RANGE': cfg.aux_range,
                                                     'AUX_kwargs': cfg.aux_kwargs}
                                         )
    if cfg.fusion_type == 'pixel':
        criterion_ahead = SetCriterion_Ahead(coeff_ssim=5.0, coeff_mse=1.0, coeff_tv=5.0, coeff_decomp=2.0,
                                     coeff_nice=0.1, coeff_cc_basic=2.0, coeff_gauss=1.0, coeff_laplace=1.0)
        criterion = criterion_ahead, criterion_crowd
        return model, criterion
    else:
        return model, criterion_crowd
