import pandas as pd
import matplotlib.pyplot as plt
import matplotlib

matplotlib.rcParams['font.family'] = 'DejaVu Sans'


def plot_distributions(csv_file: str = "applications.csv"):
    df = pd.read_csv(csv_file, encoding='utf-8-sig')

    plt.figure(figsize=(12, 6))
    city_counts = df['city'].value_counts()
    bars = plt.bar(city_counts.index, city_counts.values, color='steelblue', edgecolor='black')
    plt.title('Распределение заявок по городам', fontsize=16, fontweight='bold')
    plt.xlabel('Город', fontsize=12)
    plt.ylabel('Количество заявок', fontsize=12)
    plt.xticks(rotation=45, ha='right')

    for bar, count in zip(bars, city_counts.values):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                 str(count), ha='center', va='bottom', fontsize=10)

    threshold = len(df) * 0.4
    plt.axhline(y=threshold, color='red', linestyle='--', alpha=0.7, label=f'Порог 40% ({threshold:.0f})')
    plt.legend()

    plt.tight_layout()
    plt.savefig('cities.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Сохранено: cities.png")

    plt.figure(figsize=(12, 6))
    spec_counts = df['speciality'].value_counts()
    bars = plt.bar(spec_counts.index, spec_counts.values, color='coral', edgecolor='black')
    plt.title('Распределение заявок по специальностям', fontsize=16, fontweight='bold')
    plt.xlabel('Специальность', fontsize=12)
    plt.ylabel('Количество заявок', fontsize=12)
    plt.xticks(rotation=45, ha='right')

    for bar, count in zip(bars, spec_counts.values):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                 str(count), ha='center', va='bottom', fontsize=10)

    threshold = len(df) * 0.35
    plt.axhline(y=threshold, color='red', linestyle='--', alpha=0.7, label=f'Порог 35% ({threshold:.0f})')
    plt.legend()

    plt.tight_layout()
    plt.savefig('specialities.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Сохранено: specialities.png")

    plt.figure(figsize=(12, 6))
    df.boxplot(column='years_of_experience', by='speciality', rot=45)
    plt.title('Распределение опыта по специальностям', fontsize=14)
    plt.suptitle('')
    plt.tight_layout()
    plt.savefig('experience_by_speciality.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Сохранено: experience_by_speciality.png")


if __name__ == "__main__":
    plot_distributions()