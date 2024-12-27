import os
from pathlib import Path
from Foundation import NSURL
from CoreML import MLModel, MLModelConfiguration, MLComputePlan
from PyObjCTools import AppHelper
import enum
from colorama import Fore, Style


class ComputeDevice(enum.Enum):
    CPU = 0
    GPU = 1
    ANE = 2
    Unknown = 3

    def __str__(self):
        return self.name

    @classmethod
    def from_pyobjc(cls, device):
        from CoreML import (
            MLCPUComputeDevice,
            MLGPUComputeDevice,
            MLNeuralEngineComputeDevice,
        )

        if isinstance(device, MLCPUComputeDevice):
            return cls.CPU
        elif isinstance(device, MLGPUComputeDevice):
            return cls.GPU
        elif isinstance(device, MLNeuralEngineComputeDevice):
            return cls.ANE
        else:
            return cls.Unknown


class DeviceUsage(dict):
    def __init__(self):
        super().__init__(
            {
                ComputeDevice.CPU: 0,
                ComputeDevice.GPU: 0,
                ComputeDevice.ANE: 0,
            }
        )

    def __str__(self):
        return ", ".join(f"{device}: {count}" for device, count in self.items())


class CoreMLProfiler:
    def __init__(self, model_path: str, function_name: str = None):
        self.model_url = self._validate_and_prepare_model(model_path)
        self.compute_plan = None
        self.device_usage = None
        self.function_name = function_name

    def _validate_and_prepare_model(self, model_path: str) -> NSURL:
        """Validate the model path and convert if necessary."""
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"File {model_path} does not exist")

        compiled_path = None
        if model_path.endswith(".mlpackage"):
            if os.path.exists(model_path.replace(".mlpackage", ".mlmodelc")):
                compiled_path = model_path.replace(".mlpackage", ".mlmodelc")
            else:
                compiled_path = self._convert_mlpackage_to_mlmodelc(model_path)
        elif model_path.endswith(".mlmodelc"):
            compiled_path = model_path
        else:
            raise ValueError("Input file must be either .mlpackage or .mlmodelc")

        return NSURL.fileURLWithPath_(compiled_path)

    def _convert_mlpackage_to_mlmodelc(self, input_path) -> Path:
        """Convert .mlpackage to .mlmodelc."""
        compiled_path, error = MLModel.compileModelAtURL_error_(
            NSURL.fileURLWithPath_(input_path), None
        )
        if error:
            raise ValueError(f"Error compiling model: {error}")
        output_path = Path(input_path).with_suffix(".mlmodelc")
        Path(compiled_path).rename(output_path)
        return str(output_path)

    def _create_compute_plan(self):
        """Create a compute plan for the model."""
        config = MLModelConfiguration.alloc().init()
        MLComputePlan.loadContentsOfURL_configuration_completionHandler_(
            self.model_url, config, self._handle_compute_plan
        )
        AppHelper.runConsoleEventLoop(installInterrupt=True)

    def _handle_compute_plan(self, compute_plan, error):
        """Handle the compute plan callback."""
        if error:
            raise RuntimeError(f"Error loading compute plan: {error}")

        if compute_plan:
            self.compute_plan = compute_plan
        else:
            raise ValueError("No compute plan returned")

        AppHelper.callAfter(AppHelper.stopEventLoop)

    def list_available_functions(self):
        """List all available functions in the model."""
        if not self.compute_plan:
            self._create_compute_plan()
        
        program = self.compute_plan.modelStructure().program()
        if not program:
            return []
        
        functions = program.functions()
        return [str(key) for key in functions.allKeys()]

    def _calculate_device_usage(self) -> DeviceUsage:
        if not self.compute_plan:
            self._create_compute_plan()

        program = self.compute_plan.modelStructure().program()
        if not program:
            print("Debug: No program found")
            raise ValueError("Missing program")

        # Try to find the right function
        functions = program.functions()
        main_function = None
        
        # Debug: Print available functions
        available_functions = [str(key) for key in functions.allKeys()]
        print(f"Debug: Available functions: {available_functions}")
        
        # Try function name if specified
        if self.function_name:
            print(f"Debug: Looking for function: {self.function_name}")
            if functions.objectForKey_(self.function_name):
                main_function = functions.objectForKey_(self.function_name)
                print(f"Debug: Found requested function: {self.function_name}")
            else:
                print(f"Debug: Function {self.function_name} not found")
        
        if not main_function:
            # Try common function names
            for fname in ['main', 'predict', 'forward']:
                if functions.objectForKey_(fname):
                    main_function = functions.objectForKey_(fname)
                    print(f"Debug: Found fallback function: {fname}")
                    break
            
            # If still no function found, take the first one
            if not main_function and functions.allKeys():
                first_key = str(functions.allKeys()[0])
                main_function = functions.objectForKey_(first_key)
                print(f"Debug: Using first available function: {first_key}")
        
        if not main_function:
            print(f"Debug: No suitable function found")
            raise ValueError(f"Could not find a suitable function. Available functions: {available_functions}")

        operations = main_function.block().operations()
        print(f"Debug: Number of operations found: {len(operations)}")

        self.device_usage = DeviceUsage()
        self.operator_map = []
        op_count = 0
        for operation in operations:
            device_usage = self.compute_plan.computeDeviceUsageForMLProgramOperation_(
                operation
            )
            op_name = operation.operatorName()
            
            if device_usage:
                device_type = ComputeDevice.from_pyobjc(
                    device_usage.preferredComputeDevice()
                )
                supported_types = [ComputeDevice.from_pyobjc(d) for d in device_usage.supportedComputeDevices()]
            else:
                # Fallback device assignment based on operation type
                if op_name == 'const':
                    device_type = ComputeDevice.CPU
                    supported_types = [True, False, False]  # CPU only
                elif op_name.startswith('ios18.'):
                    device_type = ComputeDevice.ANE
                    supported_types = [True, False, True]  # CPU and ANE
                else:
                    device_type = ComputeDevice.CPU
                    supported_types = [True, True, True]  # All devices
                
            self.operator_map.append(
                {op_name: supported_types}
            )
            self.device_usage[device_type] += 1
            op_count += 1

        print(f"Debug: Total operations with device usage: {op_count}")
        return self.device_usage

    def device_usage_summary(self) -> DeviceUsage:
        """Return a summary of device usage."""
        if not self.device_usage:
            self._calculate_device_usage()
        return self.device_usage

    def operator_compatibility_report(self):
        """Return a report of operator compatibility with different compute units."""
        if not hasattr(self, 'operator_map'):
            self._calculate_device_usage()
        
        lines = []
        for op in self.operator_map:
            op_name, op_compatibility = next(iter(op.items()))
            op_compatibility = ["✅" if c else "❌" for c in op_compatibility]
            op_compatibility = "\t".join(op_compatibility)
            lines.append(f"{op_name:40}\t{op_compatibility}")
        return "\n".join(lines)

    def device_usage_summary_chart(self, total_width=50):
        """Create a bar chart representation of device counts similar to XCode."""
        if not self.device_usage:
            self._calculate_device_usage()
        total = sum(self.device_usage.values())
        title = "Compute Unit Mapping"
        bar = ""
        legend = f"All: {total}  "
        colors = {
            ComputeDevice.CPU: Fore.BLUE,
            ComputeDevice.GPU: Fore.GREEN,
            ComputeDevice.ANE: Fore.MAGENTA,
            ComputeDevice.Unknown: Fore.YELLOW,
        }

        for device, count in self.device_usage.items():
            width = int(count / total * total_width) if total > 0 else 0
            bar += colors[device] + "█" * width
            legend += f"{colors[device]}■{Style.RESET_ALL} {device}: {count}  "

        return f"\033[1m{title}\033[0m\n{bar}{Style.RESET_ALL}\n{legend}"

    def print_model_specs(self):
        """Print detailed model specifications"""
        try:
            program = self.compute_plan.modelStructure().program()
            if program:
                print("\nProgram Functions:")
                functions = program.functions()
                for key in functions.allKeys():
                    print(f"- {key}")
                
                print("\nOperations in main function:")
                main_function = functions.objectForKey_("main")
                if main_function:
                    operations = main_function.block().operations()
                    for op in operations:
                        print(f"- {op.operatorName()}")
                
        except Exception as e:
            print(f"Could not get program structure: {e}")
        
        try:
            if not self.compute_plan:
                self._create_compute_plan()
                
            model_structure = self.compute_plan.modelStructure()
            if model_structure:
                print("\nModel Structure:")
                print(f"- Input features: {model_structure.inputFeatureNames()}")
                print(f"- Output features: {model_structure.outputFeatureNames()}")
                
        except Exception as e:
            print(f"Could not get model structure: {e}")
