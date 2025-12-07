# optimization_comparison/main.py
import argparse
import matplotlib.pyplot as plt
from optimization_comparison.runner import run_all_comparisons


def parse_arguments():
    """
    Парсинг аргументов командной строки.
    """
    parser = argparse.ArgumentParser(
        description="Сравнение алгоритмов оптимизации: TA, SA и PSO",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Примеры использования:
  python -m optimization_comparison.main                # Без GIF, только графики
  python -m optimization_comparison.main --gif         # С GIF-анимациями
  python -m optimization_comparison.main --no-show     # Только сохранить графики
  python -m optimization_comparison.main --gif --no-show
  python -m optimization_comparison.main --func rastrigin --runs 10  # 10 прогонов
        """,
    )

    parser.add_argument(
        "--gif",
        action="store_true",
        default=False,
        help="Создавать GIF-анимации траекторий (по умолчанию: False)",
    )

    parser.add_argument(
        "--no-show",
        action="store_true",
        default=False,
        help="Не показывать графики интерактивно, только сохранять в файлы",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Seed для генератора случайных чисел (по умолчанию: 0)",
    )

    parser.add_argument(
        "--func",
        type=str,
        choices=["paraboloid", "rastrigin", "schwefel", "all"],
        default="all",
        help="Тестовая функция для запуска (по умолчанию: all)",
    )

    parser.add_argument(
        "--frames",
        type=int,
        default=200,
        help="Максимальное количество кадров в GIF (по умолчанию: 200)",
    )

    parser.add_argument(
        "--runs",
        type=int,
        default=1,
        help="Сколько раз повторять запуск для каждой функции (по умолчанию: 1)",
    )

    return parser.parse_args()


def run_specific_function(
    func_name: str,
    create_gif: bool = False,
    seed: int = 0,
    max_frames: int = 200,
    n_runs: int = 1,
):
    """
    Запускает сравнение на конкретной функции.
    При n_runs == 1 — один запуск.
    При n_runs > 1 — несколько запусков, берём лучшие результаты по каждому алгоритму
    и сохраняем статистику в JSON.
    """
    from optimization_comparison import benchmark_functions
    from optimization_comparison.runner import (
        run_single_comparison,
        run_repeated_single_comparison,
    )

    func_mapping = {
        "paraboloid": (
            benchmark_functions.paraboloid,
            "Параболоид f(x)=x1^2+x2^2",
            benchmark_functions.FUNCTION_BOUNDS["paraboloid"],
        ),
        "rastrigin": (
            benchmark_functions.rastrigin,
            "Функция Растригина",
            benchmark_functions.FUNCTION_BOUNDS["rastrigin"],
        ),
        "schwefel": (
            benchmark_functions.schwefel,
            "Функция Швеффеля",
            benchmark_functions.FUNCTION_BOUNDS["schwefel"],
        ),
    }

    if func_name not in func_mapping:
        raise ValueError(f"Неизвестная функция: {func_name}")

    func, func_name_str, bounds = func_mapping[func_name]

    print(f"\n{'='*50}")
    print(f"Запуск на функции: {func_name_str}")
    print(f"Seed: {seed}, GIF: {create_gif}")
    print(f"Число повторов: {n_runs}")
    print(f"{'='*50}")

    gif_filename = f"{func_name}.gif" if create_gif else None
    json_filename = f"{func_name}_stats.json" if n_runs > 1 else None

    if n_runs <= 1:
        # Обычный одиночный запуск
        fig, results = run_single_comparison(
            func=func,
            func_name=func_name_str,
            bounds=bounds,
            seed=seed,
            create_gif=create_gif,
            gif_filename=gif_filename,
            max_frames=max_frames,
        )
        stats = None
    else:
        # Пачка запусков, берём лучшие траектории + статистика
        fig, results, stats = run_repeated_single_comparison(
            func=func,
            func_name=func_name_str,
            bounds=bounds,
            n_runs=n_runs,
            base_seed=seed,
            create_gif=create_gif,
            gif_basename=func_name,
            max_frames=max_frames,
            json_filename=json_filename,
        )

    output_file = f"{func_name}_comparison.png"
    fig.savefig(output_file, dpi=150, bbox_inches="tight")
    print(f"График сохранен: {output_file}")

    if create_gif:
        print(f"GIF сохранен: {gif_filename}")

    if stats is not None and json_filename is not None:
        print(f"JSON со статистикой сохранён: {json_filename}")

    return fig, results, stats


def main():
    """
    Главная функция для запуска сравнения алгоритмов.
    """
    args = parse_arguments()

    print("Сравнение алгоритмов оптимизации")
    print("=" * 50)
    print("Параметры запуска:")
    print(f"  - Функция: {args.func}")
    print(f"  - Создавать GIF: {args.gif}")
    print(f"  - Seed: {args.seed}")
    print(f"  - Число повторов: {args.runs}")
    print(f"  - Показывать графики: {not args.no_show}")
    print(f"  - Кадров в GIF: {args.frames}")
    print("=" * 50)

    results = {}
    all_stats = {}

    if args.func == "all":
        # Запускаем все функции (с учетом количества повторов)
        results, all_stats = run_all_comparisons(
            create_gifs=args.gif,
            seed=args.seed,
            max_frames=args.frames,
            n_runs=args.runs,
        )
    else:
        # Запускаем конкретную функцию
        fig, res, stats = run_specific_function(
            func_name=args.func,
            create_gif=args.gif,
            seed=args.seed,
            max_frames=args.frames,
            n_runs=args.runs,
        )
        results[args.func] = res
        if stats is not None:
            all_stats[args.func] = stats

    # Показываем графики, если не указан флаг --no-show
    if not args.no_show:
        plt.show()

    print("\n" + "=" * 50)
    print("Сравнение завершено!")

    # Вывод списка созданных файлов
    files_created = []
    funcs_to_check = ["paraboloid", "rastrigin", "schwefel"] if args.func == "all" else [args.func]

    for func_name in funcs_to_check:
        files_created.append(f"{func_name}_comparison.png")
        if args.gif:
            files_created.append(f"{func_name}.gif")
        if args.runs > 1:
            files_created.append(f"{func_name}_stats.json")

    if args.func == "all" and args.runs > 1:
        files_created.append("all_functions_stats.json")

    print("Созданы файлы:")
    for file in files_created:
        print(f"  - {file}")


if __name__ == "__main__":
    main()