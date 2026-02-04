"""Unit tests for decorator metric collectors."""

import time
from typing import Any

import pytest
import torch

from src.benchmark.collectors.decorators import (
    track_latency_decorator,
    track_memory_decorator,
)


class TestTrackLatencyDecorator:
    """Tests for track_latency_decorator."""

    def test_decorator_registers_metrics_attribute(self):
        """Test that decorated function has metrics attribute."""

        @track_latency_decorator("test_func")
        def my_function():
            pass

        assert hasattr(my_function, "metrics")
        assert isinstance(my_function.metrics, dict)

    def test_decorator_records_latency(self):
        """Test that decorator records execution time."""

        @track_latency_decorator("inference")
        def slow_function():
            time.sleep(0.01)

        result = slow_function()

        assert "inference_latency_seconds" in slow_function.metrics
        assert slow_function.metrics["inference_latency_seconds"] >= 0.01

    def test_decorator_preserves_function_return_value(self):
        """Test that decorator preserves the function's return value."""

        @track_latency_decorator("test")
        def my_function():
            return 42

        result = my_function()
        assert result == 42

    def test_decorator_accumulates_multiple_calls(self):
        """Test that metrics are recorded after multiple calls."""

        @track_latency_decorator("repeated")
        def quick_function():
            pass

        quick_function()
        quick_function()
        quick_function()

        assert "repeated_latency_seconds" in quick_function.metrics
        assert quick_function.metrics["repeated_latency_seconds"] >= 0

    def test_decorator_with_arguments(self):
        """Test decorator works with functions that have arguments."""

        @track_latency_decorator("compute")
        def compute(x, y, multiplier=1):
            return (x + y) * multiplier

        result = compute(10, 20, multiplier=2)
        assert result == 60
        assert "compute_latency_seconds" in compute.metrics

    def test_decorator_with_kwargs(self):
        """Test decorator works with keyword arguments."""

        @track_latency_decorator("greet")
        def greet(name, greeting="Hello"):
            return f"{greeting}, {name}!"

        result = greet("World", greeting="Hi")
        assert result == "Hi, World!"
        assert "greet_latency_seconds" in greet.metrics

    def test_decorator_exception_propagation(self):
        """Test that exceptions are not swallowed by decorator."""

        @track_latency_decorator("failing")
        def failing_function():
            raise ValueError("Test error")

        with pytest.raises(ValueError):
            failing_function()

        assert "failing_latency_seconds" in failing_function.metrics

    def test_decorator_independent_metrics(self):
        """Test that different decorated functions have independent metrics."""

        @track_latency_decorator("func_a")
        def func_a():
            time.sleep(0.01)

        @track_latency_decorator("func_b")
        def func_b():
            pass

        func_a()
        func_b()

        assert "func_a_latency_seconds" in func_a.metrics
        assert "func_b_latency_seconds" in func_b.metrics
        assert "func_a_latency_seconds" not in func_b.metrics
        assert "func_b_latency_seconds" not in func_a.metrics

    def test_decorator_preserves_function_name(self):
        """Test that decorator preserves original function name."""

        @track_latency_decorator("test")
        def my_function():
            pass

        assert my_function.__name__ == "my_function"

    def test_decorator_preserves_docstring(self):
        """Test that decorator preserves original docstring."""

        @track_latency_decorator("test")
        def my_function():
            """This is my docstring."""
            pass

        assert my_function.__doc__ == "This is my docstring."

    def test_decorator_with_class_method(self):
        """Test decorator works with class methods."""

        class MyClass:
            @track_latency_decorator("method_call")
            def my_method(self):
                return "result"

        obj = MyClass()
        result = obj.my_method()

        assert result == "result"
        assert "method_call_latency_seconds" in obj.my_method.metrics

    def test_decorator_with_static_method(self):
        """Test decorator works with static methods."""

        class MyClass:
            @staticmethod
            @track_latency_decorator("static_call")
            def my_static_method():
                return "static"

        result = MyClass.my_static_method()
        assert result == "static"
        assert "static_call_latency_seconds" in MyClass.my_static_method.metrics

    def test_decorator_with_class_method(self):
        """Test decorator works with class methods."""

        class MyClass:
            @classmethod
            @track_latency_decorator("class_call")
            def my_class_method(cls):
                return cls.__name__

        result = MyClass.my_class_method()
        assert result == "MyClass"
        assert "class_call_latency_seconds" in MyClass.my_class_method.metrics


class TestTrackMemoryDecorator:
    """Tests for track_memory_decorator."""

    def test_decorator_registers_metrics_attribute(self):
        """Test that decorated function has metrics attribute."""

        @track_memory_decorator("test_func")
        def my_function():
            pass

        assert hasattr(my_function, "metrics")
        assert isinstance(my_function.metrics, dict)

    def test_decorator_records_memory_with_cuda(self):
        """Test that decorator records memory delta when CUDA is available."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        @track_memory_decorator("model_load")
        def load_tensor():
            return torch.zeros(100, 100, device="cuda")

        result = load_tensor()
        assert "model_load_memory_delta_mb" in load_tensor.metrics
        assert isinstance(load_tensor.metrics["model_load_memory_delta_mb"], int)

    def test_decorator_records_zero_memory_no_cuda(self, monkeypatch):
        """Test that decorator records zero when CUDA is unavailable."""
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

        @track_memory_decorator("test_no_cuda")
        def my_function():
            pass

        result = my_function()
        assert "test_no_cuda_memory_delta_mb" in my_function.metrics
        assert my_function.metrics["test_no_cuda_memory_delta_mb"] == 0

    def test_decorator_preserves_function_return_value(self):
        """Test that decorator preserves the function's return value."""

        @track_memory_decorator("test")
        def my_function():
            return 42

        result = my_function()
        assert result == 42

    def test_decorator_exception_propagation(self):
        """Test that exceptions are not swallowed by decorator."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        @track_memory_decorator("failing")
        def failing_function():
            tensor = torch.zeros(100, 100, device="cuda")
            raise ValueError("Test error")

        with pytest.raises(ValueError):
            failing_function()

        assert "failing_memory_delta_mb" in failing_function.metrics

    def test_decorator_independent_metrics(self):
        """Test that different decorated functions have independent metrics."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        @track_memory_decorator("func_a")
        def func_a():
            return torch.zeros(50, 50, device="cuda")

        @track_memory_decorator("func_b")
        def func_b():
            return torch.zeros(100, 100, device="cuda")

        func_a()
        func_b()

        assert "func_a_memory_delta_mb" in func_a.metrics
        assert "func_b_memory_delta_mb" in func_b.metrics
        assert "func_a_memory_delta_mb" not in func_b.metrics
        assert "func_b_memory_delta_mb" not in func_a.metrics

    def test_decorator_preserves_function_name(self):
        """Test that decorator preserves original function name."""

        @track_memory_decorator("test")
        def my_function():
            pass

        assert my_function.__name__ == "my_function"

    def test_decorator_preserves_docstring(self):
        """Test that decorator preserves original docstring."""

        @track_memory_decorator("test")
        def my_function():
            """This is my docstring."""
            pass

        assert my_function.__doc__ == "This is my docstring."

    def test_decorator_with_class_method(self):
        """Test decorator works with class methods."""

        class MyClass:
            @track_memory_decorator("method_call")
            def my_method(self):
                if torch.cuda.is_available():
                    return torch.zeros(50, 50, device="cuda")
                return None

        obj = MyClass()
        result = obj.my_method()

        assert "method_call_memory_delta_mb" in obj.my_method.metrics

    def test_decorator_with_static_method(self):
        """Test decorator works with static methods."""

        class MyClass:
            @staticmethod
            @track_memory_decorator("static_call")
            def my_static_method():
                pass

        result = MyClass.my_static_method()
        assert "static_call_memory_delta_mb" in MyClass.my_static_method.metrics

    def test_decorator_with_class_method_classmethod(self):
        """Test decorator works with class methods."""

        class MyClass:
            @classmethod
            @track_memory_decorator("class_call")
            def my_class_method(cls):
                return cls.__name__

        result = MyClass.my_class_method()
        assert result == "MyClass"
        assert "class_call_memory_delta_mb" in MyClass.my_class_method.metrics
