# MicroGradCSharp
MicroGradCSharp is an open-source project that implements a tiny scalar-valued automatic differentiation (autograd) engine and a neural network library for C# within the Unity game engine. There's nothing Unity specific in the library so you can use it for other C# projects as well.  

Inspired by Andrej Karpathy's [micrograd](https://github.com/karpathy/micrograd), this library provides a lightweight and efficient way to build and train neural networks directly in Unity. It is designed for developers and researchers looking to integrate machine learning capabilities into their Unity projects with minimal overhead. Whether you're prototyping a new game AI or experimenting with neural networks, MicroGradUnity offers a straightforward and intuitive API to get you started.

## Example usage

Supported operations (Same example as in Andrej Karpathy's [micrograd](https://github.com/karpathy/micrograd))

```csharp
Value a = new(-4.0f);

Value b = new(2.0f);

Value c = a + b;

Value d = a * b + Value.Pow(b, 3f);

c += c + 1f;

c += 1f + c + (-a);

d += d * 2f + (b + a).Relu();

d += 3f * d + (b - a).Relu();

Value e = c - d;

Value f = Value.Pow(e, 2f);

Value g = f / 2.0f;

g += 10.0f / f;

Debug.Log("Expected: 24.7041, Actual: " + g.data);

g.Backward();

//dg/da
Debug.Log("Expected: 138.8338, Actual: " + a.grad);


//dg/db
Debug.Log("Expected: 645.5773, Actual: " + b.grad);
```