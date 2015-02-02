// Copyright (c) 2015 Cranium Software

#include "TestMNIST.h"

#include "Data/MNIST.h"
#include "Layer/Layer.h"
#include "Network/AnalyticBackpropagatingNetwork.h"
#include "Network/FeedForward.h"
#include "Neuron/Input.h"
#include "Neuron/LinearNeuron.h"
#include "Neuron/Perceptron.h"
#include "Neuron/SigmoidNeuron.h"

#include <cstdio>

static const int kiTrainingRuns = 400;
static const int kiHiddenLayerSize = 40;
static const int kiOutputLayerSize = 10;
static const float kfLearningRate = 0.004f;

#define USE_OPTIMISED_NETWORK ( 0 )

void CopyInputs( float* pfFloats, unsigned char aaucPixels[ 28 ][ 28 ] )
{
    for( int i = 0; i < 28; ++i )
    {
        for( int j = 0; j < 28; ++j )
        {
            pfFloats[ j + i * 28 ] =
                static_cast< float >( aaucPixels[ i ][ j ] ) / 127.5f - 1.0f;
        }
    }
}

int TestMNIST()
{
    float afImageInputs[ 28 * 28 ] = { 0 };

#if USE_OPTIMISED_NETWORK
    const int aiSizes[] = { kiHiddenLayerSize, kiOutputLayerSize };
    AnalyticBackpropagatingNetwork< 2, 28* 28 > xNetwork( aiSizes );
#else
    // create neurons
    // create input neurons and hook up input values.
    NNL::Input* pxInputs = new NNL::Input[ 28 * 28 ];
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
#endif

    // load data
    NNL::MNIST_Image* pxTrainingImages = NNL::LoadMNISTImages( "train-images-idx3-ubyte" );
    NNL::MNIST_Label* pxTrainingLabels = NNL::LoadMNISTLabels( "train-labels-idx1-ubyte" );

    // train network
    int iCount = 0;
    xNetwork.Load();
    for( int k = 0; k < kiTrainingRuns; ++k )
    {
        for( int i = 0; i < kiMNISTTrainingSetSize; ++i )
        {
            // SE - TEMP: ...
            if( ( ( i + 1 ) % 1000 ) == 0 )
            {
                printf( "Evaluating training set %d/%d... %d correct in this batch\r\n", i + 1, kiMNISTTrainingSetSize, iCount );
                iCount = 0;
            }

            CopyInputs( afImageInputs, pxTrainingImages[ i ].maaucPixels );

            // run network
#if USE_OPTIMISED_NETWORK
            xNetwork.FeedForward( afImageInputs, NNL::Sigmoid );
#else
            xNetwork.Cycle();
#endif

            // what was the label?
            const unsigned char ucLabel = pxTrainingLabels[ i ].mucLabel;

            // back propogate
            bool bCorrect = true;
            float fMax = -FLT_MAX;
            int iMax = -1;

#if USE_OPTIMISED_NETWORK
            float afOutputs[ kiOutputLayerSize ] = { 0 };
#endif

            for( int j = 0; j < kiOutputLayerSize; ++j )
            {
#if USE_OPTIMISED_NETWORK
                afOutputs[ j ] = ( j == ucLabel ) ? 1.0f : -1.0f;
                const float fOutput = xNetwork.GetOutput( j );
#else
                const float fExpectedSignal = ( j == ucLabel ) ? 1.0f : -1.0f;
                const float fOutput = pxOutputLayer[ j ].GetResult();
#endif
                if( fOutput > fMax )
                {
                    fMax = fOutput;
                    iMax = j;
                }
                if( j == ucLabel )
                {
                    if( fOutput <= 0.0f )
                    {
                        bCorrect = false;
                    }
                }
                else if( fOutput > 0.0f )
                {
                    bCorrect = false;
                }

#if !USE_OPTIMISED_NETWORK
                pxOutputLayer[ j ].BackCycle( fExpectedSignal, kfLearningRate );
#endif
            }

            if( iMax == ucLabel )
            {
                ++iCount;
            }

#if USE_OPTIMISED_NETWORK
            xNetwork.BackPropagate( afOutputs, kfLearningRate, NNL::SigmoidDerivative );
#endif
        }

        xNetwork.Save();
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
#if USE_OPTIMISED_NETWORK
        xNetwork.FeedForward( afImageInputs, NNL::Sigmoid );
#else
        xNetwork.Cycle();
#endif

        // what was the label?
        const unsigned char ucLabel = pxTestLabels[ i ].mucLabel;
        float fMax = -FLT_MAX;
        int iMax = -1;
        for( int j = 0; j < kiOutputLayerSize; ++j )
        {
            const float fOutput =
#if USE_OPTIMISED_NETWORK
                xNetwork.GetOutput( j );
#else
                pxOutputLayer[ j ].GetResult();
#endif
            if( fOutput > fMax )
            {
                fMax = fOutput;
                iMax = j;
            }
        }
        
        const bool bCorrect = iMax == ucLabel;
        if( !bCorrect )
        {
            printf( "Guessed %d but it was %d\r\n", iMax, ucLabel );
        }
        
        iSuccessCount += bCorrect ? 1 : 0;
    }

    NNL::FreeMNISTImages( pxTestImages );
    NNL::FreeMNISTLabels( pxTestLabels );
    
#if !USE_OPTIMISED_NETWORK
    delete[] pxOutputLayer;
    delete[] pxHiddenLayer;
    delete[] pxInputs;
#endif
    
    return iSuccessCount;
}
