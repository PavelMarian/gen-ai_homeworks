from typing import List
import pandas as pd

from llm_client import make_client, get_model
from schema import Application, CITIES, SPECIALITIES, COURSES

N_APPLICATIONS = 50
MAX_RETRIES = 3

SYSTEM_PROMPT = """Ты — генератор синтетических заявок на курсы повышения квалификации (ДПО) в России.

Создай правдоподобную заявку со следующими полями:
- full_name: ФИО (русское, реальное)
- age: возраст (22-65 лет)
- address: объект с city (город из списка) и district (район города)
- speciality: текущая специальность из списка
- desired_course: желаемый курс из списка
- years_of_experience: опыт работы (0-40 лет)
- graduation_year: год окончания вуза (1980-2024)

ВАЖНО: 
1. Возраст и опыт работы должны соответствовать друг другу
2. Год окончания вуза должен быть реалистичным относительно возраста
3. Отвечай ТОЛЬКО JSON, без пояснений
"""

USER_PROMPT = "Сгенерируй одну заявку на курс повышения квалификации."

DISTRICTS_BY_CITY = {
    "Москва": ["Центральный", "Северный", "Южный", "Западный", "Восточный"],
    "Санкт-Петербург": ["Центральный", "Василеостровский", "Выборгский", "Московский"],
    "Новосибирск": ["Центральный", "Заельцовский", "Дзержинский", "Ленинский"],
    "Екатеринбург": ["Ленинский", "Октябрьский", "Кировский", "Чкаловский"],
    "Казань": ["Вахитовский", "Приволжский", "Советский", "Кировский"],
    "Нижний Новгород": ["Нижегородский", "Приокский", "Советский", "Канавинский"],
    "Красноярск": ["Центральный", "Советский", "Октябрьский", "Ленинский"],
    "Челябинск": ["Центральный", "Металлургический", "Тракторозаводский"],
    "Самара": ["Ленинский", "Октябрьский", "Куйбышевский", "Советский"],
    "Ростов-на-Дону": ["Ленинский", "Кировский", "Первомайский", "Советский"],
    "Уфа": ["Калининский", "Кировский", "Ленинский", "Октябрьский"],
    "Волгоград": ["Волгоградский", "Ворошиловский", "Дзержинский", "Краснооктябрьский"]
}

QUOTA_CITIES = list(CITIES)[:10]
QUOTA_PER_CITY = N_APPLICATIONS // len(QUOTA_CITIES)

def generate_quota_application(city: str, district: str) -> dict:
    custom_prompt = f"""Сгенерируй заявку на курс повышения квалификации.

ГОРОД: {city}
РАЙОН: {district}

Остальные поля сгенерируй самостоятельно.
Возвращай ТОЛЬКО JSON."""

    client = make_client()
    model = get_model()

    try:
        app = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": custom_prompt},
            ],
            response_model=Application,
            max_retries=MAX_RETRIES,
            temperature=0.9,
        )
        return app
    except Exception as e:
        print(f"Ошибка для {city}, {district}: {type(e).__name__}: {e}")
        return None

def generate_quota_batch() -> List[Application]:
    applications = []

    print(f"Генерация {N_APPLICATIONS} заявок с квотированием по {len(QUOTA_CITIES)} городам...")
    print(f"Квота: {QUOTA_PER_CITY} заявок на город\n")

    for city in QUOTA_CITIES:
        print(f"\n  Город: {city}")
        districts = DISTRICTS_BY_CITY.get(city, ["Центральный"])
        district_list = districts * (QUOTA_PER_CITY // len(districts) + 1)
        district_list = district_list[:QUOTA_PER_CITY]

        for i in range(QUOTA_PER_CITY):
            district = district_list[i]
            print(f"    [{i + 1}/{QUOTA_PER_CITY}] {district}...", end=" ", flush=True)

            app = generate_quota_application(city, district)
            if app:
                applications.append(app)
                print("OK")
            else:
                print("X")

    return applications

def save_to_csv(applications: List[Application], filename: str = "applications.csv"):
    data = []
    for app in applications:
        row = {
            "full_name": app.full_name,
            "age": app.age,
            "city": app.address.city,
            "district": app.address.district,
            "speciality": app.speciality,
            "desired_course": app.desired_course,
            "years_of_experience": app.years_of_experience,
            "graduation_year": app.graduation_year,
        }
        data.append(row)

    df = pd.DataFrame(data)
    df.to_csv(filename, index=False, encoding="utf-8-sig")
    print(f"\nСохранено {len(data)} заявок в {filename}")
    return df

def main():
    print("=" * 60)
    print("Генератор заявок на курсы повышения квалификации")
    print("=" * 60)

    applications = generate_quota_batch()

    df = save_to_csv(applications)

    print("\n" + "=" * 60)
    print("СТАТИСТИКА")
    print("=" * 60)
    print(f"\nВсего заявок: {len(applications)}/{N_APPLICATIONS}")

    if len(applications) < N_APPLICATIONS:
        print(f"WARNING: Сгенерировано только {len(applications)} из {N_APPLICATIONS}")

    city_dist = df['city'].value_counts()
    print(f"\nРаспределение по городам:")
    for city, count in city_dist.items():
        pct = count / len(df) * 100
        status = "OK" if pct <= 40 else "WARNING"
        print(f"  {status} {city}: {count} ({pct:.1f}%)")

    speciality_dist = df['speciality'].value_counts()
    print(f"\nРаспределение по специальностям:")
    for spec, count in speciality_dist.items():
        pct = count / len(df) * 100
        status = "OK" if pct <= 35 else "WARNING"
        print(f"  {status} {spec}: {count} ({pct:.1f}%)")

    print("\n" + "=" * 60)
    print("ПРИМЕРЫ ЗАЯВОК")
    print("=" * 60)
    for i, app in enumerate(applications[:5], 1):
        print(f"\n{i}. {app.full_name}, {app.age} лет")
        print(f"   Город: {app.address.city}, {app.address.district}")
        print(f"   Специальность: {app.speciality}")
        print(f"   Желаемый курс: {app.desired_course}")
        print(f"   Опыт: {app.years_of_experience} лет")
        print(f"   Год окончания: {app.graduation_year}")

if __name__ == "__main__":
    main()