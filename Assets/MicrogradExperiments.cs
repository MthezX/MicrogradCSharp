using Micrograd;
using System;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;


//Autograd = Automatic gradient making it easier to implement backpropagation
//Based on “The spelled-out intro to neural networks and backpropagation: building micrograd” By Anrej Karpathy
public class MicrogradExperiments : MonoBehaviour
{
    private void Start()
    {
        //ValueExperiments();

        NeuralNetworkExperiments();
    }


    private void ValueExperiments()
    {
        //Value examples
        ValueExperiments valueTest = new();

        //valueTest.TestGradients();

        //valueTest.HowDerivativesWork();

        //valueTest.BasicNeuron();
    }

    private void NeuralNetworkExperiments()
    {
        //Neural Network examples
        MicrogradNNExperiments nn = new();

        nn.XOR_Gate_Just_Values();

        //nn.XOR_Gate_NN();

        //nn.YouTube_Example();
    }
}
