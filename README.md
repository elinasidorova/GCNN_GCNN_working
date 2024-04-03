# Structure

* `Source` - all source files: models, explainers, ...
* `Experiments` - all the experiments: train files, hyperparameter search, ...
* `Notebooks` - Jupyter notebooks with visualization

## logS
Все файлы, касающиеся запуска экспериментов с logS лежат тут:

[Experiments/GCNN_FCNN/train_on_solvents](Experiments/GCNN_FCNN/train_on_solvents)

И тренировку, и оптимизацию гиперпараметров можно удобно отслеживать в MLFlow.
Для этого, после запуска обучения/оптимизации, нужно на сервере в корне репозитория вызвать
```
mlflow ui --port 8890
```
затем привязать порт 8890 сервера к порту 8890 своего компьютера 
(это можно сделать в PuTTY), и открыть в браузере адрес [localhost:8890]([http://localhost:8890])

Имена экспериментов задаются вручную при запуске и настраиваются в тех же bash-скриптах, где и прочие параметры запуска (см. ниже). 

### Train

Чтобы запустить тренировку, из корня проекта запусти

```
sbatch Experiments/GCNN_FCNN/train_on_solvents/train.sh
```

Работает в окружении `torch_geometric` на 4 сервере

Предварительно можно изменить параметры запуска (число эпох, batch size, ...) в [train.sh](Experiments/GCNN_FCNN/train_on_solvents/train.sh)

### Hyperparameters optimization

запускается так:
```
Experiments/GCNN_FCNN/train_on_solvents/optimize_hparams/optimize_hparams.sh
```

Предварительно можно модифицировать пространство поиска в
[search_space.py](Experiments/GCNN_FCNN/train_on_solvents/optimize_hparams/search_space.py),
а также параметры самого запуска в [optimize_hparams.sh](Experiments/GCNN_FCNN/train_on_solvents/optimize_hparams/optimize_hparams.sh)

Оптимизацию гиперпараметров можно отслеживать с помощью собственного визуализатора `optuna`.
Для этого на сервере в корне репозитория нужно запустить:
```
optuna-dashboard sqlite:///Output/{experiment_name}.db --port 8890
```
где вместо `{experiment_name}` следует подставить название текущего эксперимента

помимо этого нужно настроить переадресацию портов (с помощью PuTTY или аналогов),
и затем открыть в браузере адрес [localhost:8890]([http://localhost:8890])