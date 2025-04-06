# Task MaxBitSolutions 

Задача мультиклассовой классификации состояния деревьев в парках

## Install
1. Создание и активация окружения
```commandline
conda create --name mbs_trees python=3.7 && conda activate mbs_trees
```
2. Установка библиотек
```commandline
pip3 install -r requirements.txt
```
3. Установка торча 
    
   3.1 Для ГПУ версии (моя версия куды - 12.2)
    ```commandline 
    conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.7 -c pytorch -c nvidia
    ```
   3.2 Или для ЦПУ версии
    ```commandline
    conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 cpuonly -c pytorch
    ```

   
## Technical details

* API - `api.py`
* train - `train.py`
* inference - `inference.py`
* **EDA.zip** следует распаковать в корне проекта.
* Архив **resources.zip** в корне проекта, необходимо распаковать. Гит не потянул по лимитам...




## Usage
### API
1. Запусти API
```commandline
python3 api.py
```
2. Request
```commandline
curl --location 'http://127.0.0.1:8000/get_predict'	\
 --header 'Content-Type: application/json' \
 --data \
'{ 
	"tree_dbh": 2,
	"curb_loc": "OnCurb",
	"user_type": "Volunteer",
	"borough": "Brooklyn",
	"sidewalk": "NoDamage",
	"guards": "None",
	"spc_latin": "Juglans nigra",
	"steward": "None", 
	"root_stone": "No", 
	"root_grate": "No", 
	"root_other": "Yes",
	"trunk_wire": "No", 
	"trnk_light": "No", 
	"trnk_other": "No",
	"brch_light": "No", 
	"brch_shoe": "No", 
	"brch_other": "No"
}'
```
3. Answer
```commandline
{
    "Poor":0.06998120993375778,
    "Fair":0.3612702786922455,
    "Good":0.5687484741210938
}
```

### Train
1. Примеры конфиг файлов находятся в `configs`. После редактирования конфиг файла, а так же `C` списка (он использовался как конвейер экспериментов) модель можно обучать
2. Обучение модели:
```commandline
python3 train.py
```

### Inference
Пример инференса найлучшей модели, на тестовой выборке:
```commandline
python3 inference.py
```
После инференса, таблица с лейблами и предсказанными вероятностями лейблов, сохранится по `save_test_table_path` или `resources/data/processed/test_inference.csv`
 по умолчанию



## Research





## What can be improved?
