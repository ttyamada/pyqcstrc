#include <iostream>
#include <iomanip>
#include <string>
#include <vector>
#include <cmath>
#include <cassert>
using namespace std;

const double SMALL = 1.0E-30;          // used to stop divide-by-zero

using vec    = vector<double>;         // vector
using matrix = vector<vec>;            // matrix (=collection of (row) vectors)

// Function prototypes
void printMatrix( const matrix &A );
matrix matmul( const matrix &A, const matrix &B );
matrix identity( int n );
matrix transpose( const matrix &A );
bool QR( const matrix &A, matrix &Q, matrix &R );

//======================================================================


int main()
{
   matrix A = { { 12, -51,   4 },
                {  6, 167, -68 },
                { -4,  24, -41 } };              // Example in Wikipedia
   cout << "\nA:\n";   printMatrix( A  );

   matrix Q, R;

   if ( QR( A, Q, R ) )                          // Calls the QR function
   {
      cout << "\nQ: \n";   printMatrix( Q  );
      cout << "\nR: \n";   printMatrix( R  );

      cout << "\n\nCHECKS:\n";
      cout << "\nQR:\n"  ;   printMatrix( matmul( Q, R ) );
      cout << "\nQ.QT:\n";   printMatrix( matmul( Q, transpose( Q ) ) );
   }
   else
   {
      cout << "Unable to factorise\n";
   }
}


//======================================================================


void printMatrix( const matrix &A )
{
   const double NEARZERO = 1.0E-10;     // interpretation of "zero" for printing purposes
   for ( auto &row : A )
   {
      for ( auto x : row )
      {
         if ( abs( x ) < NEARZERO ) x = 0.0;
         cout << setw( 12 ) << x;
      }
      cout << '\n';
   }
}


//======================================================================


matrix matmul( const matrix &A, const matrix &B )          // Matrix times matrix
{
   int rowsA = A.size(),   colsA = A[0].size();
   int rowsB = B.size(),   colsB = B[0].size();
   assert( colsA == rowsB );

   matrix C( rowsA, vec( colsB, 0.0 ) );
   for ( int i = 0; i < rowsA; i++ )
   {
      for ( int j = 0; j < colsB; j++ )
      {
         // scalar product of ith row of A and jth column of B
         for ( int k = 0; k < colsA; k++ ) C[i][j] += A[i][k] * B[k][j];
      }
   }
   return C;
}


//======================================================================


matrix identity( int n )                                   // n x n Identity matrix
{
   matrix I( n, vec( n, 0.0 ) );
   for ( int i = 0; i < n; i++ ) I[i][i] = 1.0;
   return I;
}


//======================================================================


matrix transpose( const matrix &A )                        // Transpose
{
   int rows = A.size(),   cols = A[0].size();
   matrix AT( cols, vec( rows ) );
   for ( int i = 0; i < cols; i++ )
   {
      for ( int j = 0; j < rows; j++ ) AT[i][j] = A[j][i];
   }
   return AT;
}


//======================================================================


bool QR( const matrix &A, matrix &Q, matrix &R )
// Factorise A = QR, where Q is orthogonal, R is upper triangular
{
   int rows = A.size(), cols = A[0].size();
   if ( rows < cols )
   {
      cout << "Algorithm only works for rows >= cols\n";
      return false;
   }

   R = A;
   matrix QT = identity( rows );

   for ( int k = 0; k < cols - 1; k++ )          // k is the working column
   {
      // X vector, based on the elements from k down in the kth column
      double alpha = 0;
      for ( int i = k; i < rows; i++ ) alpha += R[i][k] * R[i][k];
      alpha = sqrt( alpha );                     // alpha is the Euclidean norm of Xk

      // V vector ( normalise Xk - alpha e_k )
      vec V( rows, 0.0 );
      double Vnorm = 0.0;
      for ( int i = k + 1; i < rows; i++ )
      {
         V[i] = R[i][k];
         Vnorm += V[i] * V[i];
      }
      V[k] = R[k][k] - alpha;
      Vnorm += V[k] * V[k];
      Vnorm = sqrt( Vnorm );
      if ( Vnorm > SMALL ) 
      {
         for ( int i = k; i < rows; i++ ) V[i] /= Vnorm;
      }

      // Householder matrix: Qk = I - 2 V VT
      matrix Qk = identity( rows );
      for ( int i = k; i < rows; i++ )
      {
         for ( int j = k; j < rows; j++ ) Qk[i][j] -= 2.0 * V[i] * V[j];
      }

      QT = matmul( Qk, QT );
      R  = matmul( Qk, R  );
   }

   Q = transpose( QT );

   return true;
}


//====================================================================== 