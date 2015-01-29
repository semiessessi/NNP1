// Copyright (c) 2015 Cranium Software

#include "TestMNIST.h"

#include "Data/MNIST.h"
#include "Layer/Layer.h"
#include "Network/FeedForward.h"
#include "Neuron/Input.h"
#include "Neuron/SigmoidNeuron.h"

#include <cstdio>

static const int kiHiddenLayerSize = 20;
static const int kiOutputLayerSize = 10;
static const float kiLearningRate = 0.1f;

void CopyInputs( float* pfFloats, unsigned char aaucPixels[ 28 ][ 28 ] )
{
    for( int i = 0; i < 28; ++i )
    {
        for( int j = 0; j < 28; ++j )
        {
            pfFloats[ j + i * 28 ] =
                static_cast< float >( aaucPixels[ i ][ j ] ) / 255.0f;
        }
    }
}

int TestMNIST()
{
    // create neurons
    // create input neurons and hook up input values.
    NNL::Input* pxInputs = new NNL::Input[ 28 * 28 ];
    float afImageInputs[ 28 * 28 ] = { 0 };
    for( int i = 0; i < 28; ++i )
    {
        for( int j = 0; j < 28; ++j )
        {
            pxInputs[ j + i * 28 ].SetInput( afImageInputs + ( j + i * 28 ) );
        }
    }
    
    // create sigmoid layer neurons
    NNL::SigmoidNeuron< 28 * 28 >* pxHiddenLayer =
        new NNL::SigmoidNeuron< 28 * 28 >[ kiHiddenLayerSize ];
    
    NNL::SigmoidNeuron< kiHiddenLayerSize >* pxOutputLayer =
        new NNL::SigmoidNeuron< kiHiddenLayerSize >[ kiOutputLayerSize ];
    
    // create layers
    NNL::Layer xInput;
    NNL::Layer xHidden;
    NNL::Layer xOutput;
    
    xInput.AddNeurons( pxInputs, 28 * 28 );
    xHidden.AddNeurons( pxHiddenLayer, kiHiddenLayerSize );
    xOutput.AddNeurons( pxOutputLayer, kiOutputLayerSize );
    
    // create network
    NNL::FeedForwardNetwork xNetwork;
    
    xNetwork.AddLayer( xInput );
    xNetwork.AddLayer( xHidden );
    xNetwork.AddLayer( xOutput );
    
    // load data
    NNL::MNIST_Image* pxTrainingImages = NNL::LoadMNISTImages( "train-images-idx3-ubyte" );
    NNL::MNIST_Label* pxTrainingLabels = NNL::LoadMNISTLabels( "train-labels-idx1-ubyte" );
    
    // train network
    for( int i = 0; i < kiMNISTTrainingSetSize; ++i )
    {
        // SE - TEMP: ...
        if( ( i % 100 ) == 0 )
        {
            printf( "Evaluating training set %d/%d...\r\n", i + 1, kiMNISTTrainingSetSize );
        }
        
        CopyInputs( afImageInputs, pxTrainingImages[ i ].maaucPixels );
        
        // run network
        xNetwork.Cycle();
        
        // what was the label?
        const unsigned char ucLabel = pxTrainingLabels[ i ].mucLabel;
        
        // back propogate
        for( int j = 0; j < kiOutputLayerSize; ++j )
        {
            pxOutputLayer[ ucLabel ].BackCycle( ( j == ucLabel ) ? 1.0f : -1.0f, kiLearningRate );
        }
    }
    
    NNL::FreeMNISTImages( pxTrainingImages );
    NNL::FreeMNISTLabels( pxTrainingLabels );
    
    // do test
    int iSuccessCount = 0;
    NNL::MNIST_Image* pxTestImages = NNL::LoadMNISTImages( "t10k-images-idx3-ubyte" );
    NNL::MNIST_Label* pxTestLabels = NNL::LoadMNISTLabels( "t10k-labels-idx1-ubyte" );
    
    for( int i = 0; i < kiMNISTTestSetSize; ++i )
    {
        // SE - TEMP: ...
        printf( "Evaluating test set %d/%d...\r\n", i, kiMNISTTestSetSize );

        
        CopyInputs( afImageInputs, pxTestImages[ i ].maaucPixels );
        
        // run network
        xNetwork.Cycle();
        
        // what was the label?
        bool bCorrect = true;
        const unsigned char ucLabel = pxTestLabels[ i ].mucLabel;
        for( int j = 0; j < kiOutputLayerSize; ++j )
        {
            if( j == ucLabel )
            {
                bCorrect = bCorrect && ( pxOutputLayer[ j ].GetResult() > 0.0f );
            }
            else
            {
                bCorrect = bCorrect && ( pxOutputLayer[ j ].GetResult() <= 0.0f );
            }
        }
        
        iSuccessCount += bCorrect ? 1 : 0;
    }

    NNL::FreeMNISTImages( pxTestImages );
    NNL::FreeMNISTLabels( pxTestLabels );
    
    delete[] pxOutputLayer;
    delete[] pxHiddenLayer;
    delete[] pxInputs;
    
    return iSuccessCount;
}
