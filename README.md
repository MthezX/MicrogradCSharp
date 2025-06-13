# MicroGradCSharp

MicroGradCSharp is an open-source project that implements a tiny scalar-valued automatic differentiation (autograd) engine and a neural network library for C# within the Unity game engine. There's nothing Unity specific in the library so you can use it for other C# projects as well.  

Inspired by Andrej Karpathy's [micrograd](https://github.com/karpathy/micrograd), this library provides a lightweight and efficient way to build and train neural networks directly in Unity. It is designed for developers and researchers looking to integrate machine learning capabilities into their Unity projects with minimal overhead. Whether you're prototyping a new game AI or experimenting with neural networks, MicroGradUnity offers a straightforward and intuitive API to get you started.

> [!CAUTION]
> Bla bla bla Skynet bla bla bla own risk


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


## Neural Network

A common "Hello World" example when making Neural Networks is the [XOR gate](https://en.wikipedia.org/wiki/XOR_gate). We want to create a Neural Network that understands the following:

| Input x1 | Input x2 | Output y |
| ---------| -------- | -------- |
| 0        | 0        | 0        |
| 0        | 1        | 1        |
| 1        | 0        | 1        |
| 1        | 1        | 0        |

Init the neural network:

```csharp
MicroMath.Random.Seed(0);

//Training data XOR
float[][] inputDataFloat = { new[] { 0f, 0f }, new[] { 0f, 1f }, new[] { 1f, 0f }, new[] { 1f, 1f } };
float[] outputDataFloat = new[] { 0f, 1f, 1f, 0f };

//Convert training data from float to Value
Value[][] inputData = Array.ConvertAll(inputDataFloat, subArray => Array.ConvertAll(subArray, item => new Value(item)));
Value[] outputData = Array.ConvertAll(outputDataFloat, item => new Value(item));

//How fast/slow the network will learn
float learningRate = 0.1f;

//Create the NN
//2 inputs, 3 neurons in the middle layer, 1 output
MLP nn = new(2, new int[] { 3, 1 });

TrainNN(nn, learningRate, inputData, outputData);
TestNN(nn, inputData, outputData);
```

Train the neural network:

```csharp
for (int i = 0; i <= 100; i++)
{
    //Forward pass

    //Catch all outputs for this batch
    Value[] networkOutputs = new Value[outputData.Length];

    for (int inputDataIndex = 0; inputDataIndex < outputData.Length; inputDataIndex++)
    {
        //Run input through the network
        Value[] outputArray = nn.Activate(inputData[inputDataIndex]);

        //We know we have just a single output
        Value output = outputArray[0];

        networkOutputs[inputDataIndex] = output;
    }

    //Error calculations using MSE
    Value loss = new(0f);

    for (int j = 0; j < networkOutputs.Length; j++)
    {
        Value wantedOutput = outputData[j];
        Value actualOutput = networkOutputs[j];

        Value errorSquare = Value.Pow(actualOutput - wantedOutput, 2f);

        loss += errorSquare;
    }

    if (i % 10 == 0)
    {
        Debug.Log($"Iteration: {i}, Network error: {loss.data}");
    }

    //Backward pass
	
	//First reset the gradients
    nn.ZeroGrad();

    //Calculate the gradients
    loss.Backward();

    //Optimize the weights and biases by using gradient descent
    Value[] parameters = nn.GetParameters();

    foreach (Value param in parameters)
    {
        param.data += -learningRate * param.grad;
    }
}
```

Test the neural network:

```csharp
for (int inputDataIndex = 0; inputDataIndex < outputData.Length; inputDataIndex++)
{
    Value[] outputArray = nn.Activate(inputData[inputDataIndex]);

    float wantedData = outputData[inputDataIndex].data;
    float actualData = outputArray[0].data;

    Debug.Log("Wanted: " + wantedData + ", Actual: " + actualData);
}
```

When I ran the neural network I got the following results:

| Input x1 | Input x2 | Output y |
| ---------| -------- | -------- |
| 0        | 0        | 0,022296 |
| 0        | 1        | 0,959968 |
| 1        | 0        | 0,961034 |
| 1        | 1        | 0,026877 |

The outputs are very close to the 0 and 1 we wanted - the output will never be exactly 0 or 1. 
