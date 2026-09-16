#pragma once
#include <iostream>
#include <vector>
#include <array>
#include <cmath>
#include <numeric>

#include "crsys.hpp"

//using namespace std;

// implementation
namespace qnnum {    
    int N;

    class Qnnum { 
    public:
        //array<int, 3> n;
        int n[3];
        int N;
    
        int signm(const int a) const {
            if (a>=0) {
                return 1;
            } else {
                return -1;
            }
        }
    
        Qnnum() {
            N=crsys::N;
            n[0]=0;
            n[1]=0;
            n[2]=1;
        } // default constructor
    
        Qnnum (int n0, int n1, int n2) {
            N=crsys::N;
            n[0]=n0; n[1]=n1; n[2]=n2;
        }  // construtor

        Qnnum(int n_[]) {
            N=crsys::N;
            for (int i=0; i<3; ++i) {
                n[i] = n_[i];
            }
        }  // constructor
    
        Qnnum operator+(const Qnnum& b) const {
            Qnnum a=*this;
            int c1 = a.n[0] * b.n[2] + b.n[0] * a.n[2];
            int c2 = a.n[1] * b.n[2] + b.n[1] * a.n[2];
            int c3 = a.n[2] * b.n[2];
            if (c3==0 && c1>0) return Qnnum(1,0,0);  // inf
            if (c3==0 && c1<0) return Qnnum(-1,0,0); // -inf
            if (c3==0 && c1==0) return Qnnum(0,0,0);  // nan
            //int g = (int)gcd(c1, (int)gcd(c2, c3));
            int g = std::gcd(c1, std::gcd(c2, c3));
            if (c3<0) {
                int d1[] = {-c1 / g, -c2 / g, -c3 / g}; return Qnnum(d1);
            } else {
                int d1[] = {c1 / g, c2 / g, c3 / g}; return Qnnum(d1);
            }
            //int d1[] = {c1 / g, c2 / g, c3 / g};
            //return Qnnum(d1);
        }
    
        Qnnum operator-(const Qnnum& b) const {
            Qnnum a=*this;
            int c1 = a.n[0] * b.n[2] - b.n[0] * a.n[2];
            int c2 = a.n[1] * b.n[2] - b.n[1] * a.n[2];
            int c3 = a.n[2] * b.n[2];
            if (c3==0 && c1>0) return Qnnum(1,0,0);  // inf
            if (c3==0 && c1<0) return Qnnum(-1,0,0); // -inf
            if (c3==0 && c1==0) return Qnnum(0,0,0);  // nan
            //std::cout<<"c1 "<<c1<<" c2 "<<c2<<" c3 "<<c3<<std::endl;  // for test
            //int g = (int)gcd(c1, (int)gcd(c2, c3));
            int g = std::gcd(c1, std::gcd(c2, c3));
            if (c3<0) {
                int d1[] = {-c1 / g, -c2 / g, -c3 / g}; return Qnnum(d1);
            } else {
                int d1[] = {c1 / g, c2 / g, c3 / g}; return Qnnum(d1);
            }
            //int d1[] = {c1 / g, c2 / g, c3 / g};
            //return Qnnum(d1);
        }
    
        Qnnum operator+=(const Qnnum& b) {
            Qnnum a=*this;
            return a + b;
        }
    
        Qnnum operator-=(const Qnnum& b) {
            Qnnum a=*this;
            return a - b;
        }
    
        Qnnum operator*(const Qnnum& b) const {
            Qnnum a=*this;
            //std::cout<<"N "<<N<<std::endl;  // for test
            int c1 = a.n[0] * b.n[0] + a.n[1] * b.n[1] * N;
            int c2 = a.n[0] * b.n[1] + a.n[1] * b.n[0];
            int c3 = a.n[2] * b.n[2];
            if (c3==0 && c1>0) return Qnnum(1,0,0);  // inf
            if (c3==0 && c1<0) return Qnnum(-1,0,0); // -inf
            if (c3==0 && c1==0) return Qnnum(0,0,0);  // nan
            int g = std::gcd(c1, std::gcd(c2, c3));
            if (c3<0) {
                int d1[] = {-c1 / g, -c2 / g, -c3 / g}; return Qnnum(d1);
            } else {
                int d1[] = {c1 / g, c2 / g, c3 / g};  return Qnnum(d1);
            }
        }
    
        Qnnum operator*(const int b) const {
            Qnnum a=*this;
            int c1 = a.n[0] * b;
            int c2 = a.n[1] * b;
            int c3 = a.n[2];
            if (c3==0 && c1>0) return Qnnum(1,0,0);  // inf
            if (c3==0 && c1<0) return Qnnum(-1,0,0); // -inf
            if (c3==0 && c1==0) return Qnnum(0,0,0);  // nan
            int g = std::gcd(c1, std::gcd(c2, c3));
            if (c3<0) {
                int d1[] = {-c1 / g, -c2 / g, -c3 / g}; return Qnnum(d1);
            } else {
                int d1[] = {c1 / g, c2 / g, c3 / g}; return Qnnum(d1);
            }
        }
    
        Qnnum operator/(const Qnnum& b) const {
            Qnnum a=*this;
            // c = 1/b
            if (b == Qnnum(0,0,1)) {
                // if this>0 return infinity else this<0 return -infinity
                if (a > Qnnum(0,0,1)) {
                    return Qnnum(1,0,0);  // inf
                } else if (a < Qnnum(0,0,1)) {
                    return Qnnum(-1,0,0); // -inf
                } else if (a.n[2] == 0) {  // inf/inf = nan
                    return Qnnum(0,0,0);  // nan
                }
            }

            //std::cout<<"N "<<N<<std::endl; // for test
            int c1 = b.n[0] * b.n[2];
            int c2 = -b.n[1] * b.n[2];
            int c3 = b.n[0] * b.n[0] - b.n[1] * b.n[1] * N;
            if (c3==0 && c1>0) return Qnnum(1,0,0);  // inf
            if (c3==0 && c1<0) return Qnnum(-1,0,0); // -inf
            if (c3==0 && c1==0) return Qnnum(0,0,0);  // nan
            int d1[] = {c1, c2, c3};
            Qnnum qn = Qnnum(d1);
            return a * qn;
        }

    
        Qnnum operator/(const int b) const {
            Qnnum a=*this;
            int d1[] = {a.n[0], a.n[1], a.n[2] * b};
            return Qnnum(d1);
        }
    
        bool operator==(const Qnnum& b) const {
            Qnnum a=*this;
            //Qnnum c = a - b;
            //return (c.n[0] == 0 && c.n[1] == 0);
            return (a.n[0] == b.n[0] && a.n[1] == b.n[1] && a.n[2] == b.n[2]);
        }

        bool operator==(const int b_) const {
            Qnnum a=*this;
            int bt[3]={b_,0,1};
            Qnnum b=Qnnum(bt);
            return (a == b);
            //Qnnum c = a - b;
            //return (a.n[0] == b.n[0] && a.n[1] == b.n[1] && a.n[2] == b.n[2]);
        }

        bool operator!=(const Qnnum& b) const {
            Qnnum a=*this;
            return !(a == b);
            /**
            if (a==Qnnum(1,0,0) || a==Qnnum(-1,0,0)) {
                if (b==a) {
                    return false;
                } else {
                    return true;
                }
            }
            Qnnum c = a - b;
            return (c.n[0] != 0 || c.n[1] != 0);
            **/
        }

        bool operator!=(const int b_) const {
            Qnnum a=*this;
            int bt[3]={b_,0,1};
            Qnnum b=Qnnum(bt);
            //Qnnum c = a - b;
            //return (c.n[0] != 0 || c.n[1] != 0);
            return !(a==b);
        }
    
        bool operator<(const Qnnum& b) const {
            Qnnum a=*this;
            //std::cout<<"N "<<N<<std::endl; // for test
            if (a==Qnnum(1,0,0)) return false;
            if (a==-Qnnum(1,0,0)) return true;
            Qnnum c = a - b;
            return (signm(c.n[0]) * c.n[0] * c.n[0] + signm(c.n[1]) * c.n[1] * c.n[1] * N < 0);
        }

        bool operator<(const int b_) const {
            Qnnum a=*this;
            int bt[3]={b_,0,1};
            Qnnum b=Qnnum(bt);
            if (a==Qnnum(1,0,0)) return false;
            if (a==-Qnnum(1,0,0)) return true;
            //std::cout<<"N "<<N<<std::endl; // for test
            Qnnum c = a - b;
            return (signm(c.n[0]) * c.n[0] * c.n[0] + signm(c.n[1]) * c.n[1] * c.n[1] * N < 0);
        }

        bool operator<=(const Qnnum& b) const {
            Qnnum a=*this;
            return (a == b || a < b);
            /**
            //std::cout<<"N "<<N<<std::endl; // for test
            Qnnum c = a - b;
            return (signm(c.n[0]) * c.n[0] * c.n[0] + signm(c.n[1]) * c.n[1] * c.n[1] * N <= 0);
            **/
        }

        bool operator<=(const int b_) const {
            Qnnum a=*this;
            int bt[3]={b_,0,1};
            Qnnum b=Qnnum(bt);
            return (a == b || a < b);
            //std::cout<<"N "<<N<<std::endl; // for test
            //Qnnum c = a - b;
            //return (signm(c.n[0]) * c.n[0] * c.n[0] + signm(c.n[1]) * c.n[1] * c.n[1] * N <= 0);
        }
     
        bool operator>(const Qnnum& b) const {
            Qnnum a=*this;
            if (a==Qnnum(1,0,0)) return true;
            if (a==-Qnnum(1,0,0)) return false;
            //std::cout<<"N "<<N<<std::endl; // for test
            Qnnum c = a - b;
            return (signm(c.n[0]) * c.n[0] * c.n[0] + signm(c.n[1]) * c.n[1] * c.n[1] * N > 0);
        }

        bool operator>(const int b_) const {
            Qnnum a=*this;
            int bt[3]={b_,0,1};
            Qnnum b=Qnnum(bt);
            if (a==Qnnum(1,0,0)) return true;
            if (a==-Qnnum(1,0,0)) return false;
            //std::cout<<"N "<<N<<std::endl; // for test
            Qnnum c = a - b;
            return (signm(c.n[0]) * c.n[0] * c.n[0] + signm(c.n[1]) * c.n[1] * c.n[1] * N > 0);
        }
    
        bool operator>=(const Qnnum& b) const {
            Qnnum a=*this;
            return (a == b || a > b);
            //std::cout<<"N "<<N<<std::endl; // for test
            //Qnnum c = a - b;
            //return (signm(c.n[0]) * c.n[0] * c.n[0] + signm(c.n[1]) * c.n[1] * c.n[1] * N >= 0);
        }

        bool operator>=(const int b_) const {
            Qnnum a=*this;
            int bt[3]={b_,0,1};
            Qnnum b=Qnnum(bt);
            return (a >= b);
            //std::cout<<"N "<<N<<std::endl; // for test
            //Qnnum c = a - b;
            //return (signm(c.n[0]) * c.n[0] * c.n[0] + signm(c.n[1]) * c.n[1] * c.n[1] * N >= 0);
        }
    
        Qnnum operator-() {
            Qnnum a=*this;
            int d1[] = {-a.n[0], -a.n[1], a.n[2]};
            return Qnnum(d1);
        }    

        friend std::ostream& operator<<(std::ostream& os, const Qnnum& a) {
            // Write obj to stream
            os << " [" << a.n[0] << " " << a.n[1] << " " << a.n[2] << "] ";
            return os;
        }
    };

    void qnnum_init() {
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

    Qnnum inf() {
        //int infty = static_cast<int>(std::pow(2, 20));  // infty=2^20
        int d1[] = {1,0,0};  // inf (infinity)
        return Qnnum(d1);
    }

    Qnnum nan() {
        //int infty = static_cast<int>(std::pow(2, 20));  // infty=2^20
        int d1[] = {0,0,0};  // nan (nonumber)
        return Qnnum(d1);
    }

    Qnnum any(int  n_[]) {
        int d1[] = {n_[0],n_[1],n_[2]};
        return Qnnum(d1);
    }

    Qnnum any(int  n0, int n1, int n2) {
        int d1[] = {n0, n1, n2};
        return Qnnum(d1);
    }

    Qnnum neg(Qnnum& a) {
        int d1[]={-a.n[0],-a.n[1],a.n[2]};
        return Qnnum(d1);
    }

    /*
    Qnnum operator/(int a_, const Qnnum& b) const {
        int d1[] = {a_, 0, 1};
        Qnnum a= Qnnum(d1);
        return a / b;
    }
    */

    Qnnum abs(Qnnum& a) {
        if (a<0) {
            int d1[]={-a.n[0],-a.n[1],a.n[2]};
            return Qnnum(d1);
        }
        return a;
    }

    Qnnum min(Qnnum& a, Qnnum& b) {
        if (a<b) {
            return a;
        } else {
            return b;
        }
    }

    Qnnum max(Qnnum& a, Qnnum& b) {
        if (a>b) {
            return a;
        } else {
            return b;
        }
    }

    bool is_inf(Qnnum& a) {
        return a==inf();
    }

    double qn2flt(Qnnum& a) {
        return (a.n[0]+a.n[1]*sqrt((float)N))/a.n[2];
    }

    Qnnum int2qn(int i, int N) {
        int d1[] = {i,0,1};
        return Qnnum(d1);
    }

    // float to Qnnum converter
    Qnnum flt2qn(float qr) {
        float xm=std::abs(qr);
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
                            if(std::abs(xd) < xm) {
                                float xm=std::abs(xd);
                                xn.n[0]=n1;
                                xn.n[1]=n2;
                                xn.n[2]=n3;
                            }
                            if(std::abs(xd)<eps) {
                                //print("xt,n1,n2,n3.sqrtnr",xt,n1,n2,n3,sqrtn)
                                //printqnn("xn",xn)
                                return xn;
                            }
                        }
                    }
                }
            }
        }
        std::cout << "cannt convert float to Qnnum" << std::endl;
        return zero();
    }


    // printqnn
    void printqnn(const std::string& str, Qnnum& a) {
        std::cout << str << " [" << a.n[0] << " " << a.n[1] << " " << a.n[2] << "]" << std::endl;
    }

    void printqnn_nlf(const std::string& str, Qnnum& a) {
        std::cout << str << " [" << a.n[0] << " " << a.n[1] << " " << a.n[2] << "] ";
    }

    void printqnns(const std::string& str, std::vector<Qnnum>& a) {
        for (size_t i = 0; i < a.size(); ++i) {
            std::cout << str << " [" << a[i].n[0] << " " << a[i].n[1] << " " << a[i].n[2] << "]" << std::endl;
        }
    }
} // namespace
    

