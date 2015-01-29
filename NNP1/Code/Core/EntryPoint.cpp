// Copyright (c) 2015 Cranium Software

#include "Layer/Layer.h"
#include "Maths/Random.h"
#include "Neuron/Constant.h"
#include "Neuron/Input.h"
#include "Neuron/Perceptron.h"
#include "Network/FeedForward.h"

float TestFunctionAnd( const float fX, const float fY )
{
    return 2.0f * static_cast< float >( ( ( fX < 0.0f ) ? 1 : 0 ) & ( ( fY < 0.0f ) ? 1 : 0 ) ) - 1.0f;
}

float TestFunctionOr( const float fX, const float fY )
{
    return 2.0f * static_cast< float >( ( ( fX < 0.0f ) ? 1 : 0 ) | ( ( fY < 0.0f ) ? 1 : 0 ) ) - 1.0f;
}

float TestFunctionNand( const float fX, const float fY )
{
    return 2.0f * static_cast< float >( !( ( ( fX < 0.0f ) ? 1 : 0 ) & ( ( fY < 0.0f ) ? 1 : 0 ) ) ) - 1.0f;
}

float TestFunctionXor( const float fX, const float fY )
{
    return 2.0f * static_cast< float >( ( ( fX < 0.0f ) ? 1 : 0 ) ^ ( ( fY < 0.0f ) ? 1 : 0 ) ) - 1.0f;
}

float TestFunctionOne( const float, const float )
{
    return 1.0f;
}

int main( const int, const char* const* const )
{
    float fInX = 0.0f;
    float fInY = 1.0f;
    NNL::Input xInX( &fInX );
    NNL::Input xInY( &fInY );
    NNL::Perceptron< 2 > xPerceptronA;
    NNL::Perceptron< 2 > xPerceptronB;
    NNL::Perceptron< 2 > xPerceptronC;
    NNL::Perceptron< 3 > xPerceptronD;
    NNL::Layer xLayer1;
    NNL::Layer xLayer2;
    NNL::Layer xLayer3;
    NNL::FeedForwardNetwork xNetwork;

    xLayer1.AddNeuron( xInX );
    xLayer1.AddNeuron( xInY );

    xLayer2.AddNeuron( xPerceptronA );
    xLayer2.AddNeuron( xPerceptronB );
    xLayer2.AddNeuron( xPerceptronC );

    xLayer3.AddNeuron( xPerceptronD );

    xNetwork.AddLayer( xLayer1 );
    xNetwork.AddLayer( xLayer2 );
    xNetwork.AddLayer( xLayer3 );

    const int kiIterations = 100000;
    const int kiHalf = kiIterations >> 1;
    const float kfLearningRate = 0.1f;

    {
        int iRight = 0;
        for( int i = 0; i < kiIterations; ++i )
        {
            fInX = ( ( NNL::WeakRandomInt() & 0xFF ) > 0x7F ) ? 1.0f : -1.0f;
            fInY = ( ( NNL::WeakRandomInt() & 0xFF ) > 0x7F ) ? 1.0f : -1.0f;
            xNetwork.Cycle( );
            const float fTestResult = TestFunctionOne( fInX, fInY );
            xNetwork.BackCycle( fTestResult, kfLearningRate );
            //printf( "%f %f\r\n", xPerceptron.GetResult(), fTestResult );

            if( xPerceptronD.GetResult() == fTestResult )
            {
                ++iRight;
            }
            else
            {
                iRight = 0;
            }
        }

        if( iRight > kiHalf )
        {
            printf( "Learned 1 after %d iterations\r\n", kiIterations - iRight );
        }
        else
        {
            printf( "Failed to learn 1!\r\n" );
        }
    }

    {
        int iRight = 0;
        for( int i = 0; i < kiIterations; ++i )
        {
            fInX = ( ( NNL::WeakRandomInt() & 0xFF ) > 0x7F ) ? 1.0f : -1.0f;
            fInY = ( ( NNL::WeakRandomInt() & 0xFF ) > 0x7F ) ? 1.0f : -1.0f;
            xNetwork.Cycle();
            const float fTestResult = TestFunctionAnd( fInX, fInY );
            xNetwork.BackCycle( fTestResult, kfLearningRate );
            //printf( "%f %f\r\n", xPerceptron.GetResult(), fTestResult );

            if( xPerceptronD.GetResult() == fTestResult )
            {
                ++iRight;
            }
            else
            {
                iRight = 0;
            }
        }

        if( iRight > kiHalf )
        {
            printf( "Learned AND after %d iterations\r\n", kiIterations - iRight );
        }
        else
        {
            printf( "Failed to learn AND!\r\n" );
        }
    }

    {
        int iRight = 0;
        for( int i = 0; i < kiIterations; ++i )
        {
            fInX = ( ( NNL::WeakRandomInt() & 0xFF ) > 0x7F ) ? 1.0f : -1.0f;
            fInY = ( ( NNL::WeakRandomInt() & 0xFF ) > 0x7F ) ? 1.0f : -1.0f;
            xNetwork.Cycle();
            const float fTestResult = TestFunctionOr( fInX, fInY );
            xNetwork.BackCycle( fTestResult, kfLearningRate );
            //printf( "%f %f\r\n", xPerceptron.GetResult(), fTestResult );

            if( xPerceptronD.GetResult() == fTestResult )
            {
                ++iRight;
            }
            else
            {
                iRight = 0;
            }
        }

        if( iRight > kiHalf )
        {
            printf( "Learned OR after %d iterations\r\n", kiIterations - iRight );
        }
        else
        {
            printf( "Failed to learn OR!\r\n" );
        }
    }

    {
        int iRight = 0;
        for( int i = 0; i < kiIterations; ++i )
        {
            fInX = ( ( NNL::WeakRandomInt() & 0xFF ) > 0x7F ) ? 1.0f : -1.0f;
            fInY = ( ( NNL::WeakRandomInt() & 0xFF ) > 0x7F ) ? 1.0f : -1.0f;
            xNetwork.Cycle();
            const float fTestResult = TestFunctionNand( fInX, fInY );
            xNetwork.BackCycle( fTestResult, kfLearningRate );
            //printf( "%f %f\r\n", xPerceptron.GetResult(), fTestResult );

            if( xPerceptronD.GetResult() == fTestResult )
            {
                ++iRight;
            }
            else
            {
                iRight = 0;
            }
        }

        if( iRight > kiHalf )
        {
            printf( "Learned NAND after %d iterations\r\n", kiIterations - iRight );
        }
        else
        {
            printf( "Failed to learn NAND!\r\n" );
        }
    }

    {
        int iRight = 0;
        for( int i = 0; i < kiIterations; ++i )
        {
            fInX = ( ( NNL::WeakRandomInt() & 0xFF ) > 0x7F ) ? 1.0f : -1.0f;
            fInY = ( ( NNL::WeakRandomInt() & 0xFF ) > 0x7F ) ? 1.0f : -1.0f;
            xNetwork.Cycle( );
            const float fTestResult = TestFunctionXor( fInX, fInY );
            xNetwork.BackCycle( fTestResult, kfLearningRate );
            //printf( "%f %f\r\n", xPerceptron.GetResult(), fTestResult );

            if( xPerceptronD.GetResult() == fTestResult )
            {
                ++iRight;
            }
            else
            {
                iRight = 0;
            }
        }

        if( iRight > kiHalf )
        {
            printf( "Learned XOR after %d iterations\r\n", kiIterations - iRight );
        }
        else
        {
            printf( "Failed to learn XOR!\r\n" );
        }
    }

    return 0;
}
