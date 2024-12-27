import argparse
from .prof import CoreMLProfiler
import sys

def main():
    parser = argparse.ArgumentParser(description='CoreML Model Profiler')
    parser.add_argument('model_path', help='Path to .mlpackage or .mlmodelc file')
    parser.add_argument('--report', '-r', 
                       choices=['usage', 'chart', 'operators', 'all', 'specs'],
                       default='chart', 
                       help='Type of report to generate')
    parser.add_argument('--list-functions', '-l', action='store_true',
                       help='List available functions in the model')
    parser.add_argument('--function', '-f', help='Specify which function to analyze')

    args = parser.parse_args()
    
    try:
        profiler = CoreMLProfiler(args.model_path, args.function)
        
        if args.list_functions:
            print("\nAvailable functions:")
            print("\n".join(profiler.list_available_functions()))
            return
        
        # Calculate device usage first if needed for operators report
        if args.report in ['operators', 'all']:
            profiler.device_usage_summary()  # This ensures operator_map is populated
        
        if args.report == 'specs' or args.report == 'all':
            print("\nModel Specifications:")
            profiler.print_model_specs()
        
        if args.report == 'usage' or args.report == 'all':
            print("\nDevice Usage Summary:")
            print(profiler.device_usage_summary())
        
        if args.report == 'chart' or args.report == 'all':
            print("\nDevice Usage Chart:")
            print(profiler.device_usage_summary_chart())
        
        if args.report == 'operators' or args.report == 'all':
            print("\nOperator Compatibility Report:")
            print(profiler.operator_compatibility_report())
            
    except FileNotFoundError as e:
        print(f"\nError: {e}")
        print("\nPlease check the file path and make sure the model exists.")
        sys.exit(1)
    except Exception as e:
        print(f"\nError: An unexpected error occurred: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main() 