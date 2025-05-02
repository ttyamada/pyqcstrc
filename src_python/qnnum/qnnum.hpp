#pragma once
#include <array>
#include <numeric>
#include <cmath>
#include <iostream>
#include "crsys.hpp"

using namespace std;

namespace qnnum {

int N;

class Qnnum { 

    public:
    //array<int, 3> n;
    int n[3];

    int signm(int a) {
        if (a>=0) {
            return 1;
        } else {
            return -1;
        }
    }

    //Qnnum(array<int, 3>& n_) {
    Qnnum(int n_[]) {
        for (int i=0; i<3; ++i) {
            n[i] = n_[i];
        }
    }

    Qnnum operator+(Qnnum& b) {
        Qnnum a=*this;
        int c1 = a.n[0] * b.n[2] + b.n[0] * a.n[2];
        int c2 = a.n[1] * b.n[2] + b.n[1] * a.n[2];
        int c3 = a.n[2] * b.n[2];
        //int g = (int)gcd(c1, (int)gcd(c2, c3));
        int g = gcd(c1, gcd(c2, c3));
        if (c3<0) {
            int d1[] = {-c1 / g, -c2 / g, -c3 / g}; return Qnnum(d1);
        } else {
            int d1[] = {c1 / g, c2 / g, c3 / g}; return Qnnum(d1);
        }
        //int d1[] = {c1 / g, c2 / g, c3 / g};
        //return Qnnum(d1);
    }

    Qnnum operator-(Qnnum& b) {
        Qnnum a=*this;
        int c1 = a.n[0] * b.n[2] - b.n[0] * a.n[2];
        int c2 = a.n[1] * b.n[2] - b.n[1] * a.n[2];
        int c3 = a.n[2] * b.n[2];
        //int g = (int)gcd(c1, (int)gcd(c2, c3));
        int g = gcd(c1, gcd(c2, c3));
        if (c3<0) {
            int d1[] = {-c1 / g, -c2 / g, -c3 / g}; return Qnnum(d1);
        } else {
            int d1[] = {c1 / g, c2 / g, c3 / g}; return Qnnum(d1);
        }
        //int d1[] = {c1 / g, c2 / g, c3 / g};
        //return Qnnum(d1);
    }

    Qnnum operator+=(Qnnum& b) {
        Qnnum a=*this;
        return a + b;
    }

    Qnnum operator-=(Qnnum& b) {
        Qnnum a=*this;
        return a - b;
    }

    Qnnum operator*(Qnnum& b) {
        Qnnum a=*this;
        //cout<<"N "<<N<<endl;  // for test
        int c1 = a.n[0] * b.n[0] + a.n[1] * b.n[1] * N;
        int c2 = a.n[0] * b.n[1] + a.n[1] * b.n[0];
        int c3 = a.n[2] * b.n[2];
        int g = gcd(c1, gcd(c2, c3));
        if (c3<0) {
            int d1[] = {-c1 / g, -c2 / g, -c3 / g}; return Qnnum(d1);
        } else {
            int d1[] = {c1 / g, c2 / g, c3 / g};  return Qnnum(d1);
        }
    }

    Qnnum operator*(const int b) {
        Qnnum a=*this;
        int c1 = a.n[0] * b;
        int c2 = a.n[1] * b;
        int c3 = a.n[2];
        int g = gcd(c1, gcd(c2, c3));
        if (c3<0) {
            int d1[] = {-c1 / g, -c2 / g, -c3 / g}; return Qnnum(d1);
        } else {
            int d1[] = {c1 / g, c2 / g, c3 / g}; return Qnnum(d1);
        }
        
    }

    Qnnum operator/(Qnnum& b) {
        Qnnum a=*this;
        //cout<<"N "<<N<<endl; // for test
        int c1 = b.n[0] * b.n[2];
        int c2 = -b.n[1] * b.n[2];
        int c3 = b.n[0] * b.n[0] - b.n[1] * b.n[1] * N;
        if (c3 == 0) throw runtime_error("ERROR: division by zero");
        int d1[] = {c1, c2, c3};
        Qnnum qn = Qnnum(d1);
        return a * qn;
    }

    Qnnum operator/(int b) {
        Qnnum a=*this;
        int d1[] = {a.n[0], a.n[1], a.n[2] * b};
        return Qnnum(d1);
    }

    bool operator==(Qnnum& b) {
        Qnnum a=*this;
        Qnnum c = a - b;
        return (c.n[0] == 0 && c.n[1] == 0);
    }

    bool operator<(Qnnum& b) {
        Qnnum a=*this;
        //cout<<"N "<<N<<endl; // for test
        Qnnum c = a - b;
        return (signm(c.n[0]) * c.n[0] * c.n[0] + signm(c.n[1]) * c.n[1] * c.n[1] * N < 0);
    }

    bool operator>(Qnnum& b) {
        Qnnum a=*this;
        //cout<<"N "<<N<<endl; // for test
        Qnnum c = a - b;
        return (signm(c.n[0]) * c.n[0] * c.n[0] + signm(c.n[1]) * c.n[1] * c.n[1] * N > 0);
    }

    Qnnum operator-() {
        Qnnum a=*this;
        int d1[] = {-a.n[0], -a.n[1], a.n[2]};
        return Qnnum(d1);
    }
};

//namespace qnnum {

// Function implementations
void init() {
    N=crsys::N;
}

Qnnum zero() {
    int d1[] = {0, 0, 1};
    return Qnnum(d1);
}

Qnnum one() {
    int d1[] = {1,0,1};
    return Qnnum(d1);
}

Qnnum any(int  n_[]) {
    int d1[] = {n_[0],n_[1],n_[2]};
    return Qnnum(d1);
}

/*
Qnnum abs(Qnnum& a) {
    if (a<0) {
        return neg(a);
    }
    return a;
}
*/

double qn2flt(Qnnum& a) {
    return (a.n[0]+a.n[1]*sqrt((float)N))/a.n[2];
}

Qnnum int2qn(int i, int N) {
    int d1[] = {i,0,1};
    return Qnnum(d1);
}

// float to Qnnum converter
Qnnum flt2qn(float qr) {
    float xm=abs(qr);
    int isg[]={1,-1};
    float sqrtn=sqrt(float(N));
    //print("N",N,"sqrtn",sqrtn) // for test
    float eps=0.000001;
    int n1m=200; int n2m=200; int n3m=200;
    int d1[] = {0,0,1};
    Qnnum xn=Qnnum(d1);
    for (int k=0; k<n3m; ++k) {
        int n3=k+1;
        for (int i=0; i<n1m; ++i) {
            for (int j=0; j<n2m; ++j) {
                for (int ic=0; ic<2; ++ic) {
                    int n1=isg[ic]*i; //+-i
                    for (int jc=0; jc<2; ++jc) {
                        int n2=isg[jc]*j; //+-j
                        float xt=(n1+n2*sqrtn)/n3;
                        //print("xt",xt,"qr",qr)
                        float xd=(xt-qr);
                        if(abs(xd) < xm) {
                            float xm=abs(xd);
                            xn.n[0]=n1;
                            xn.n[1]=n2;
                            xn.n[2]=n3;
                        }
                        if(abs(xd)<eps) {
                            //print("xt,n1,n2,n3.sqrtnr",xt,n1,n2,n3,sqrtn)
                            //printqnn("xn",xn)
                            return xn;
                        }
                    }
                }
            }
        }
    }
    cout << "cannt convert float to qnnum" << endl;
    return zero();
}


// printqnn
void printqnn(const string& str, Qnnum& a) {
    cout << str << " [" << a.n[0] << " " << a.n[1] << " " << a.n[2] << "]" << endl;
}

void printqnns(const string& str, vector<Qnnum>& a) {
    for (size_t i = 0; i < a.size(); ++i) {
        cout << str << " [" << a[i].n[0] << " " << a[i].n[1] << " " << a[i].n[2] << "]" << endl;
    }
}

};
