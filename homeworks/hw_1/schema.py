from datetime import datetime
from typing import Literal
from pydantic import BaseModel, Field, field_validator

CITIES = {
    "Москва", "Санкт-Петербург", "Новосибирск", "Екатеринбург",
    "Казань", "Нижний Новгород", "Красноярск", "Челябинск",
    "Самара", "Ростов-на-Дону", "Уфа", "Волгоград"
}

SPECIALITIES = Literal[
    "менеджер", "инженер", "бухгалтер", "учитель",
    "врач", "юрист", "IT-специалист", "маркетолог",
    "продавец-консультант", "администратор"
]

COURSES = Literal[
    "Управление проектами",
    "Анализ данных в Excel",
    "Корпоративные финансы",
    "Продвинутый английский",
    "Цифровой маркетинг",
    "HR-менеджмент",
    "Python для анализа данных",
    "Управление командой"
]


class Address(BaseModel):
    city: str
    district: str = Field(min_length=2, max_length=40)

    @field_validator("city")
    @classmethod
    def city_must_be_in_list(cls, v: str) -> str:
        if v not in CITIES:
            raise ValueError(f"Город «{v}» не из утверждённого списка")
        return v


class Application(BaseModel):
    full_name: str = Field(min_length=2, max_length=100)
    age: int = Field(ge=22, le=65)
    address: Address
    speciality: SPECIALITIES
    desired_course: COURSES
    years_of_experience: int = Field(ge=0, le=40)
    graduation_year: int = Field(ge=1980, le=2024)

    @field_validator("graduation_year")
    @classmethod
    def validate_graduation_consistency(cls, v: int, info) -> int:
        current_year = datetime.now().year
        age = info.data.get('age')

        if age is None:
            return v

        max_possible_graduation = current_year + age - 22

        if v > max_possible_graduation:
            raise ValueError(
                f"Несоответствие: возраст {age} лет, а год окончания {v} — "
                f"это слишком поздно (максимум {max_possible_graduation})"
            )

        min_possible_graduation = current_year + age - 65

        if v < min_possible_graduation and age > 22:
            raise ValueError(
                f"Несоответствие: возраст {age} лет, а год окончания {v} — "
                f"это слишком рано (минимум {min_possible_graduation})"
            )

        return v

    @property
    def city(self) -> str:
        return self.address.city
