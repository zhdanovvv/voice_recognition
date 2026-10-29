Модель распознавания пола диктора. В файле female.txt перечислены женские голоса, в файле male.txt мужские голоса. В папке wav_data/train_data аудио файлы, для обучения модели, 
в папке wav_data/test_data, для проверки. Все файлы папки wav_data в формате PCM 8000 Гц, 16 бит.

Установка зависимостей: pip install -r requirements.txt

Запуск: 
1. Перейдите в директорию Py_files: cd Gender_identification/Py_files/
2. Создайте виртуальное окружение: python3 -m venv gender_identification_environment
3. Активируйте виртуальное окружение: source gender_identification_environment/bin/activate
4. Установите зависимости: pip install -r requirements.txt
5. Запустите программу: python3 test_gender.py
6. Чтобы выйти из виртуального окружения, деактивируйте его: deactivate
