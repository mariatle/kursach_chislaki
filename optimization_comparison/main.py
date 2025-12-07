# optimization_comparison/main.py
import argparse
import matplotlib.pyplot as plt
from optimization_comparison.runner import run_all_comparisons


def parse_arguments():
    """
    Парсинг аргументов командной строки.
    """
    parser = argparse.ArgumentParser(
        description='Сравнение алгоритмов оптимизации: TA, SA и PSO',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Примеры использования:
  python -m optimization_comparison.main                # Без GIF, только графики
  python -m optimization_comparison.main --gif         # С GIF-анимациями
  python -m optimization_comparison.main --no-show     # Только сохранить графики
  python -m optimization_comparison.main --gif --no-show
        """
    )
    
    parser.add_argument(
        '--gif',
        action='store_true',
        default=False,
        help='Создавать GIF-анимации траекторий (по умолчанию: False)'
    )
    
    parser.add_argument(
        '--no-show',
        action='store_true',
        default=False,
        help='Не показывать графики интерактивно, только сохранять в файлы'
    )
    
    parser.add_argument(
        '--seed',
        type=int,
        default=0,
        help='Seed для генератора случайных чисел (по умолчанию: 0)'
    )
    
    parser.add_argument(
        '--func',
        type=str,
        choices=['paraboloid', 'rastrigin', 'schwefel', 'all'],
        default='all',
        help='Тестовая функция для запуска (по умолчанию: all)'
    )
    
    parser.add_argument(
        '--frames',
        type=int,
        default=200,
        help='Максимальное количество кадров в GIF (по умолчанию: 200)'
    )
    
    return parser.parse_args()


def run_specific_function(func_name: str, create_gif: bool = False, seed: int = 0, max_frames: int = 200):
    """
    Запускает сравнение на конкретной функции.
    """
    from optimization_comparison import benchmark_functions
    from optimization_comparison.runner import run_single_comparison
    
    func_mapping = {
        'paraboloid': (
            benchmark_functions.paraboloid,
            "Параболоид f(x)=x1^2+x2^2",
            benchmark_functions.FUNCTION_BOUNDS["paraboloid"]
        ),
        'rastrigin': (
            benchmark_functions.rastrigin,
            "Функция Растригина",
            benchmark_functions.FUNCTION_BOUNDS["rastrigin"]
        ),
        'schwefel': (
            benchmark_functions.schwefel,
            "Функция Швеффеля",
            benchmark_functions.FUNCTION_BOUNDS["schwefel"]
        ),
    }
    
    if func_name not in func_mapping:
        raise ValueError(f"Неизвестная функция: {func_name}")
    
    func, func_name_str, bounds = func_mapping[func_name]
    
    print(f"\n{'='*50}")
    print(f"Запуск на функции: {func_name_str}")
    print(f"Seed: {seed}, GIF: {create_gif}")
    
    gif_filename = f"{func_name}.gif" if create_gif else None
    
    fig, results = run_single_comparison(
        func=func,
        func_name=func_name_str,
        bounds=bounds,
        seed=seed,
        create_gif=create_gif,
        gif_filename=gif_filename,
    )
    
    output_file = f"{func_name}_comparison.png"
    fig.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"График сохранен: {output_file}")
    
    if create_gif:
        print(f"GIF сохранен: {gif_filename}")
    
    return fig, results


def main():
    """
    Главная функция для запуска сравнения алгоритмов.
    """
    args = parse_arguments()
    
    print("Сравнение алгоритмов оптимизации")
    print("="*50)
    print(f"Параметры запуска:")
    print(f"  - Функция: {args.func}")
    print(f"  - Создавать GIF: {args.gif}")
    print(f"  - Seed: {args.seed}")
    print(f"  - Показывать графики: {not args.no_show}")
    print(f"  - Кадров в GIF: {args.frames}")
    print("="*50)
    
    results = {}
    
    if args.func == 'all':
        # Запускаем все функции
        results = run_all_comparisons(create_gifs=args.gif)
    else:
        # Запускаем конкретную функцию
        fig, res = run_specific_function(
            func_name=args.func,
            create_gif=args.gif,
            seed=args.seed,
            max_frames=args.frames
        )
        results[args.func] = res
    
    # Показываем графики, если не указан флаг --no-show
    if not args.no_show:
        plt.show()
    
    print("\n" + "="*50)
    print("Сравнение завершено!")
    
    # Вывод списка созданных файлов
    files_created = []
    funcs_to_check = ['paraboloid', 'rastrigin', 'schwefel'] if args.func == 'all' else [args.func]
    
    for func_name in funcs_to_check:
        files_created.append(f"{func_name}_comparison.png")
        if args.gif:
            files_created.append(f"{func_name}.gif")
    
    print("Созданы файлы:")
    for file in files_created:
        print(f"  - {file}")


if __name__ == "__main__":
    main()