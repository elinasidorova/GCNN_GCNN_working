import optuna

from Source.models.FCNN.optimize_hparams import FCNNParams
from Source.models.GCNN.optimize_hparams import GCNNParams, GeneralParams as GCNNGeneralParams


class GeneralParams(GCNNGeneralParams):
    def __init__(self, trial: optuna.Trial, metal_featurizer_variants, *args, **kwargs):
        super(GeneralParams, self).__init__(trial, *args, **kwargs)
        self.metal_featurizer_variants = metal_featurizer_variants

    def get_metal_featurizer(self):
        metal_featurizer_name = self.trial.suggest_categorical("metal_featurizer_name",
                                                               list(self.metal_featurizer_variants.keys()))
        metal_featurizer = self.metal_featurizer_variants[metal_featurizer_name]
        return metal_featurizer

    def get(self):
        result = super().get()
        result.update({
            "metal_featurizer": self.get_metal_featurizer()
        })
        return result


class GCNNFCNNParams:
    def __init__(self, trial: optuna.Trial, global_pooling_variants, metal_fc_params, gcnn_params, post_fc_params):
        self.trial = trial
        self.metal_fc_params = FCNNParams(self.trial, **metal_fc_params, prefix="metal_FCNN").get()
        self.gcnn_params = GCNNParams(self.trial, **gcnn_params, prefix="GCNN").get()
        self.post_fc_params = FCNNParams(self.trial, **post_fc_params, prefix="post_FCNN").get()
        self.global_pooling_variants = global_pooling_variants
        self.global_pooling_name = None

    def get_global_pooling(self):
        self.global_pooling_name = self.trial.suggest_categorical(
            "global_pooling", list(self.global_pooling_variants.keys())
        )
        return self.global_pooling_variants[self.global_pooling_name]

    def get(self):
        model_parameters = {
            "metal_fc_params": self.metal_fc_params,
            "gcnn_params": self.gcnn_params,
            "post_fc_params": self.post_fc_params,
            "global_pooling": self.get_global_pooling(),
        }

        return model_parameters
