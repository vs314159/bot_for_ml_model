Цей бот - обгортка для моделі, що передбачає вартість оренди житла в Києві на основі даних, зібраних за квітень (і трохи за травень) 2023 року
Для користування ботом, потрібно редагувати credentials.yml
url можна отримати, наприклад з ngrok 
ngrok http 5005 - далі скопіювати посилання, що з'явиться
Далі в терміналі: rasa run actions - це дасть змогу боту використовувати кастомні дії, такі як передбачення вартості і запуск заново опитування
і у другому терміналі (паралельно): rasa run

модель передбачення ціни можна змінити в ml_model, тоді і в predictor.py відповідно потрібно буде зчитати нову модель
