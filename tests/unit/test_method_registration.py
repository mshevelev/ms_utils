"""Unit tests for the method_registration module.

Tests cover the CustomNamespace descriptor, NamespaceInstance wrapper,
and the register_method decorator functionality.
"""

import pytest
from ms_utils.method_registration import (
    CustomNamespace,
    NamespaceInstance,
    register_method,
)


class DummyClass:
    """Test class for method registration."""

    def __init__(self, value):
        self.value = value


class AnotherDummyClass:
    """Another test class for multi-class registration."""

    def __init__(self, data):
        self.data = data


class TestCustomNamespace:
    """Tests for CustomNamespace descriptor."""

    def test_namespace_creation(self):
        """Test creating a CustomNamespace."""
        ns = CustomNamespace(name="test")
        assert ns._name == "test"

    def test_class_level_access_returns_descriptor(self):
        """Test that accessing namespace from class returns the descriptor itself."""
        DummyClass.custom = CustomNamespace(name="custom")
        result = DummyClass.custom
        assert isinstance(result, CustomNamespace)
        assert result._name == "custom"

    def test_instance_level_access_returns_namespace_instance(self):
        """Test that accessing namespace from instance returns NamespaceInstance."""
        DummyClass.custom = CustomNamespace(name="custom")
        obj = DummyClass(42)
        result = obj.custom
        assert isinstance(result, NamespaceInstance)
        assert result._instance is obj


class TestNamespaceInstance:
    """Tests for NamespaceInstance wrapper."""

    def test_namespace_instance_creation(self):
        """Test creating a NamespaceInstance."""
        ns = CustomNamespace(name="test")
        obj = DummyClass(42)
        ns_instance = NamespaceInstance(ns, obj)

        assert ns_instance._namespace is ns
        assert ns_instance._instance is obj

    def test_method_binding(self):
        """Test that methods are properly bound to the instance."""
        # Create a namespace and add a method to it
        ns = CustomNamespace(name="custom")

        def test_method(obj):
            return obj.value * 2

        setattr(ns, "double", test_method)

        # Create instance and access through NamespaceInstance
        obj = DummyClass(21)
        ns_instance = NamespaceInstance(ns, obj)

        # Call the method - should automatically pass obj as first argument
        result = ns_instance.double()
        assert result == 42

    def test_method_with_additional_args(self):
        """Test that additional arguments are passed correctly."""
        ns = CustomNamespace(name="custom")

        def add_method(obj, x, y):
            return obj.value + x + y

        setattr(ns, "add", add_method)

        obj = DummyClass(10)
        ns_instance = NamespaceInstance(ns, obj)

        result = ns_instance.add(5, 3)
        assert result == 18

    def test_method_with_kwargs(self):
        """Test that keyword arguments are passed correctly."""
        ns = CustomNamespace(name="custom")

        def format_method(obj, prefix="", suffix=""):
            return f"{prefix}{obj.value}{suffix}"

        setattr(ns, "format", format_method)

        obj = DummyClass("test")
        ns_instance = NamespaceInstance(ns, obj)

        result = ns_instance.format(prefix=">>", suffix="<<")
        assert result == ">>test<<"

    def test_dir_returns_namespace_methods(self):
        """Test that __dir__ returns the methods from the namespace."""
        ns = CustomNamespace(name="custom")

        def method1(obj):
            pass

        def method2(obj):
            pass

        setattr(ns, "method1", method1)
        setattr(ns, "method2", method2)

        obj = DummyClass(42)
        ns_instance = NamespaceInstance(ns, obj)

        methods = dir(ns_instance)
        assert "method1" in methods
        assert "method2" in methods


class TestRegisterMethod:
    """Tests for the register_method decorator."""

    def test_register_single_method_single_class(self):
        """Test registering a single method on a single class."""

        class TestClass1:
            def __init__(self, val):
                self.val = val

        @register_method([TestClass1], namespace="ops")
        def square(obj):
            return obj.val**2

        # Verify namespace exists
        assert hasattr(TestClass1, "ops")

        # Verify method works
        obj = TestClass1(5)
        assert obj.ops.square() == 25

    def test_register_multiple_methods_same_namespace(self):
        """Test registering multiple methods in the same namespace."""

        class TestClass2:
            def __init__(self, val):
                self.val = val

        @register_method([TestClass2], namespace="math")
        def double(obj):
            return obj.val * 2

        @register_method([TestClass2], namespace="math")
        def triple(obj):
            return obj.val * 3

        obj = TestClass2(7)
        assert obj.math.double() == 14
        assert obj.math.triple() == 21

    def test_register_same_method_multiple_classes(self):
        """Test registering the same method on multiple classes."""

        class ClassA:
            def __init__(self, x):
                self.x = x

        class ClassB:
            def __init__(self, x):
                self.x = x

        @register_method([ClassA, ClassB], namespace="utils")
        def increment(obj):
            return obj.x + 1

        a = ClassA(10)
        b = ClassB(20)

        assert a.utils.increment() == 11
        assert b.utils.increment() == 21

    def test_register_with_accessors_attribute(self):
        """Test that _accessors set is updated when present."""

        class ClassWithAccessors:
            _accessors = set()

            def __init__(self, val):
                self.val = val

        @register_method([ClassWithAccessors], namespace="custom")
        def test_method(obj):
            return obj.val

        assert "custom" in ClassWithAccessors._accessors

    def test_method_can_access_instance_attributes(self):
        """Test that registered methods can access instance attributes."""

        class Person:
            def __init__(self, first_name, last_name):
                self.first_name = first_name
                self.last_name = last_name

        @register_method([Person], namespace="name")
        def full_name(obj):
            return f"{obj.first_name} {obj.last_name}"

        person = Person("John", "Doe")
        assert person.name.full_name() == "John Doe"

    def test_method_can_modify_instance(self):
        """Test that registered methods can modify instance state."""

        class Counter:
            def __init__(self):
                self.count = 0

        @register_method([Counter], namespace="ops")
        def increment(obj):
            obj.count += 1
            return obj.count

        counter = Counter()
        assert counter.count == 0
        assert counter.ops.increment() == 1
        assert counter.count == 1

    def test_namespace_on_different_instances(self):
        """Test that namespace methods work independently on different instances."""

        class Box:
            def __init__(self, value):
                self.value = value

        @register_method([Box], namespace="transform")
        def negate(obj):
            return -obj.value

        box1 = Box(10)
        box2 = Box(-5)

        assert box1.transform.negate() == -10
        assert box2.transform.negate() == 5

    def test_chaining_with_instance_methods(self):
        """Test that registered methods can be chained with regular instance methods."""

        class Calculator:
            def __init__(self, value):
                self.value = value

            def add(self, x):
                self.value += x
                return self

        @register_method([Calculator], namespace="ops")
        def get_value(obj):
            return obj.value

        calc = Calculator(10)
        result = calc.add(5).add(3).ops.get_value()
        assert result == 18


class TestEdgeCases:
    """Tests for edge cases and special scenarios."""

    def test_multiple_namespaces_on_same_class(self):
        """Test that a class can have multiple different namespaces."""

        class MultiNamespace:
            def __init__(self, val):
                self.val = val

        @register_method([MultiNamespace], namespace="math")
        def double(obj):
            return obj.val * 2

        @register_method([MultiNamespace], namespace="string")
        def to_str(obj):
            return str(obj.val)

        obj = MultiNamespace(42)
        assert obj.math.double() == 84
        assert obj.string.to_str() == "42"

    def test_namespace_persists_across_instances(self):
        """Test that namespace registration persists at the class level."""

        class Persistent:
            def __init__(self, val):
                self.val = val

        @register_method([Persistent], namespace="calc")
        def triple(obj):
            return obj.val * 3

        # Create instance after registration
        obj1 = Persistent(5)
        assert obj1.calc.triple() == 15

        # Create another instance - should still have the namespace
        obj2 = Persistent(7)
        assert obj2.calc.triple() == 21

    def test_registering_to_existing_namespace(self):
        """Test adding methods to a namespace that already exists."""

        class Existing:
            def __init__(self, val):
                self.val = val

        # First registration creates the namespace
        @register_method([Existing], namespace="ops")
        def method1(obj):
            return obj.val + 1

        # Second registration should add to existing namespace
        @register_method([Existing], namespace="ops")
        def method2(obj):
            return obj.val - 1

        obj = Existing(10)
        assert obj.ops.method1() == 11
        assert obj.ops.method2() == 9


class TestIntegrationScenarios:
    """Integration tests simulating real-world usage patterns."""

    def test_dataframe_like_extension(self):
        """Test a pandas DataFrame-like extension pattern."""

        class DataFrame:
            _accessors = set()

            def __init__(self, data):
                self.data = data

            def shape(self):
                return (len(self.data), len(self.data[0]) if self.data else 0)

        @register_method([DataFrame], namespace="custom")
        def row_count(df):
            """Get number of rows."""
            return df.shape()[0]

        @register_method([DataFrame], namespace="custom")
        def col_count(df):
            """Get number of columns."""
            return df.shape()[1]

        df = DataFrame([[1, 2, 3], [4, 5, 6]])

        # Verify _accessors was updated
        assert "custom" in DataFrame._accessors

        # Verify methods work
        assert df.custom.row_count() == 2
        assert df.custom.col_count() == 3

    def test_series_and_dataframe_shared_namespace(self):
        """Test registering methods that work on both Series and DataFrame."""

        class Series:
            def __init__(self, data):
                self.data = data

        class DataFrame:
            def __init__(self, data):
                self.data = data

        @register_method([Series, DataFrame], namespace="stats")
        def describe(obj):
            return f"Data type: {type(obj).__name__}, Length: {len(obj.data)}"

        series = Series([1, 2, 3, 4])
        df = DataFrame([[1, 2], [3, 4]])

        assert "Series" in series.stats.describe()
        assert "DataFrame" in df.stats.describe()
