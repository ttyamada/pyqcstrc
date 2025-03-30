module fmath   ! extended version of math
  implicit none
  !private

  interface chsgn
    module procedure chsgnr,chsgnd
  end interface

  interface add3vctr
    module procedure add3vctrr,add3vctrd
  end interface

  interface mtxvec
    module procedure mtxvecr,mtxvecd,mtxvecir,mtxvecid,mtxveci,mtxvecrd,mtxvecdr
  end interface mtxvec

  interface vecxmt
    module procedure vecxmti,vecxmtr,vecxmtd,vecxmtri,vecxmtrd,vecxmtdr,vecxmtdi,vecxmtir,vecxmtid
  end interface vecxmt

  interface matinv
    module procedure matinvr,matinvd
  end interface

  interface wtvec
    module procedure wtvctrr,wtvctri
    module procedure wtvctrrf,wtvctrif
  end interface wtvec

  interface wtmtrx
    module procedure wtmtrxr,wtmtrxd,wtmtrxi
    module procedure wtmtrxrf,wtmtrxdf,wtmtrxif
  end interface wtmtrx

  interface zerov
    module procedure zerovr,zerovi,zerovc,zerovd
  end interface zerov

  interface mcpy3
    module procedure mcpyi3,mcpyr3,mcpyd3,mcpydr3
  end interface mcpy3

  interface mcpy6
    module procedure mcpyi6,mcpyr6,mcpyir6,mcpyri6,mcpyd6,mcpyid6,mcpydr6
  end interface mcpy6

  interface mcpy
    module procedure mcpyi,mcpyr,mcpyd,mcpyir,mcpyid
  end interface mcpy

  interface vinp
    module procedure vinpi,vinpr,vinpd,vinpir,vinpri,vinpc,vinpz
  end interface vinp

  interface matinv3
    module procedure matinv3r,matinv3d
  end interface matinv3

  interface matinv6
    module procedure matinv6r,matinv6d
  end interface matinv6

  interface mtmpl
    module procedure mtmplr,mtmpli,mtmplir,mtmplri,mtmpld,mtmpldr
     !module procedure rctmtmpl
  end interface mtmpl

  interface sclvec
    module procedure sclvecr
  end interface sclvec

  interface sclmtrx
    module procedure sclmtrxr,sclmtrxd,sclmtrxdr
  end interface

  interface iszerov
    module procedure iszerovr0,iszerovr1,iszerovi
  end interface iszerov

  interface mttransp  ! transpose matrix
    module procedure mttranspr,mttranspi !,mttranspr,mttranspi
  end interface mttransp

  interface rctmttransp  ! transpose matrix
    module procedure rctmttranspr
  end interface rctmttransp

  interface atanf
    module procedure atanfr
  end interface

  interface mtmplmno
    module procedure mtmplmnor,mtmplmnoi,mtmplmnoir,mtmplmnori
    module procedure mtmplmnod,mtmplmnoid,mtmplmnodi
    module procedure mtmplmnor1n,mtmplmnoi1n,mtmplmnoir1n,mtmplmnori1n
    module procedure mtmplmnorn1,mtmplmnoin1,mtmplmnoirn1,mtmplmnorin1
  end interface mtmplmno

  !interface mtxvec
  !   module procedure mtxvecqt1,mtxvecqt2,mtxvecqt3
  !end interface

  interface extrrctmtrx
    module procedure extrrrctmtrx
  end interface extrrctmtrx

  interface wtrctmtrx
    module procedure wtrrctmtrx
  end interface wtrctmtrx

  interface vnegate
    module procedure vnegater,vnegatei
  end interface vnegate

  interface vprd   ! vector pvcroduct of 3D vectors
    module procedure  vprdr,vprdi
  end interface vprd

  interface vcpy
    module procedure vcpyr,vcpyi,vcpyd
  end interface vcpy

  interface negate
    module procedure negater,negatei
  end interface negate

  interface determ3
    module procedure determ3r,determ3r1
  end interface determ3

  interface vsum
    module procedure vsumr,vsumi,vsumir
  end interface vsum

  interface vsub
    module procedure vsubr,vsubi
  end interface vsub

  interface vabs
    module procedure vabsr,vabsd
  end interface

  interface norml
    module procedure normlr,normld
  end interface

  interface mtrnsp
    module procedure trnspr,trnspd,trnspi
  end interface mtrnsp

  interface mtrnsp3
    module procedure trnspr3,trnspi3,trnspd3
  end interface mtrnsp3

  interface mtrnsp6
    module procedure trnspr6,trnspi6,trnspd6
  end interface mtrnsp6

  interface mtmpl3
    module procedure mtmplr3,mtmpli3
  end interface mtmpl3

  interface mtmpl6
    module procedure mtmplr6,mtmpli6
  end interface mtmpl6

  interface diagmtrx
    module procedure diagmtrxr,diagmtrxi,diagmtrxd
  end interface diagmtrx

  interface unitmtrx
    module procedure unitmtrxr,unitmtrxi,unitmtrxd
  end interface unitmtrx

  interface qsort
    module procedure qsortr,qsortr1,qsorti,qsorti1
  end interface qsort

  interface setrowmtrx
    module procedure setrowmtrxr,setrowmtrxi
  end interface setrowmtrx

  interface getrowmtrx
    module procedure getrowmtrxr,getrowmtrxi
  end interface getrowmtrx

  interface wtmtrx6
    module procedure wtrmtrx6,wtimtrx6
  end interface wtmtrx6

  interface wtmtrx3
    module procedure wtrmtrx3,wtimtrx3
  end interface wtmtrx3

  !interface wtvec6
  !    module procedure wtrvct6
  !end interface

  interface zeromtrx
    module procedure zeromtrxr,zeromtrxi,zeromtrxd
  end interface zeromtrx

!  public :: mtxvec
!  public ::  getpi
!  public :: vinp
!  public ::  vabs
!  public :: iszerov
!  public ::  atanf
!  public ::  determ3r
!  public ::  lcm
!  public ::  gcd
!  public ::  add3vctr
!  public :: mcpy6
!  public :: mtmpl
!  public :: mtmpl3
!  public :: mtmpl6
!  public :: mtmplmno
!  public :: mtrnsp
!  public :: mtrnsp3
!  public :: mtrnsp6
!  public :: qsort
!  public :: mcpy
!  public :: mcpy3
!  public ::  zerov
!  public ::  vfact
!  public ::  vmult
!  public ::  vsclr
!  public :: sclvec
!  public :: getrowmtrx
!  public :: setrowmtrx
!  public ::  sclmtrx
!  public :: vcpy
!  public :: negate
!  public :: vsum
!  public :: vsub
!  public ::  vprd
!  public ::  chsgn
!  public :: vnegate
!  public :: diagmtrx
!  public :: unitmtrx
!  public :: zeromtrx
!  public ::  norml
!  public :: wtvec
!  public :: wtmtrx
!  public :: matinv3
!  public :: matinv6
!  public ::  matinvr
!  public ::  GAUSSJ
!  !public :: wtvec6
!  public :: wtmtrx6
!  public :: wtmtrx3
!  public :: wtrctmtrx
!  public :: extrrctmtrx
!  public :: rctmttransp

contains

  real(4) function getpi()
    implicit none
    real(4) pi
    !real(4) getpi
    pi=atan2(1.0,1.0)*4
    getpi=pi
    return
  end function getpi

  ! add three vectors
  subroutine add3vctrr(v1,v2,v3,v,n)
    implicit none
    integer:: n
    real(4) :: v1(:)
    real(4) :: v2(:)
    real(4) :: v3(:)
    real(4) :: v(:)
    integer :: i
    call zerovr(v,n)
    do i=1,n
      v(i)=v1(i)+v2(i)+v3(i)
    end do
    return
  end subroutine add3vctrr

    ! add three vectors
  subroutine add3vctrd(v1,v2,v3,v,n)
    implicit none
    integer:: n
    real(8) :: v1(:)
    real(8) :: v2(:)
    real(8) :: v3(:)
    real(8) :: v(:)
    integer :: i
    call zerovd(v,n)
    do i=1,n
      v(i)=v1(i)+v2(i)+v3(i)
    end do
    return
  end subroutine add3vctrd

  ! copy real 6x6 matrix
  subroutine mcpyr6(from,to)
    implicit none
    real(4) :: from(6,6)
    real(4) :: to(6,6)
    integer :: i
    integer :: j
    do i=1,6
      do j=1,6
        to(i,j)=from(i,j)
      end do
    end do
    return
  end subroutine mcpyr6

  ! copy real 6x6 matrix
  subroutine mcpydr6(from,to)
    implicit none
    real(8) :: from(6,6)
    real(4) :: to(6,6)
    integer :: i
    integer :: j
    do i=1,6
      do j=1,6
        to(i,j)=from(i,j)
      end do
    end do
    return
  end subroutine mcpydr6

    ! copy real 6x6 matrix
  subroutine mcpyd6(from,to)
    implicit none
    real(8) :: from(6,6)
    real(8) :: to(6,6)
    integer :: i
    integer :: j
    do i=1,6
      do j=1,6
        to(i,j)=from(i,j)
      end do
    end do
    return
  end subroutine mcpyd6

  ! copy integer 6x6 matrix
  subroutine mcpyi6(from,to)
    implicit none
    integer :: from(6,6)
    integer :: to(6,6)
    integer :: i
    integer :: j
    do i=1,6
      do j=1,6
        to(i,j)=from(i,j)
      end do
    end do
    return
  end subroutine mcpyi6

  ! copy integer 6x6 to real 6x6 matrix
  subroutine mcpyir6(from,to)
    implicit none
    integer :: from(6,6)
    real(4) :: to(6,6)
    integer :: i
    integer :: j
    do i=1,6
      do j=1,6
        to(i,j)=from(i,j)
      end do
    end do
    return
  end subroutine mcpyir6

    ! copy real 6x6 to integer 6x6 matrix
  subroutine mcpyri6(from,to)
    implicit none
    real(4) :: from(6,6)
    integer :: to(6,6)
    integer :: i
    integer :: j
    do i=1,6
      do j=1,6
        to(i,j)=from(i,j)
      end do
    end do
    return
  end subroutine mcpyri6

    ! copy integer 6x6 to real 6x6 matrix
  subroutine mcpyid6(from,to)
    implicit none
    integer :: from(6,6)
    real(8) :: to(6,6)
    integer :: i
    integer :: j
    do i=1,6
      do j=1,6
        to(i,j)=from(i,j)
      end do
    end do
    return
  end subroutine mcpyid6

  ! multiply real nxn matrices
  subroutine mtmplr(a,b,c,n)
    implicit none
    integer :: n
    real(4) :: a(n,n)
    real(4) :: b(n,n)
    real(4) :: c(n,n)
    integer :: i
    integer :: j
    integer :: k
    do i=1,n
      do j=1,n
        c(i,j)=0.
        do k=1,n
          c(i,j)=c(i,j)+a(i,k)*b(k,j)
        end do
      end do
    end do
    return
  end subroutine mtmplr

    ! multiply real nxn matrices
  subroutine mtmpld(a,b,c,n)
    implicit none
    integer :: n
    real(8) :: a(n,n)
    real(8) :: b(n,n)
    real(8) :: c(n,n)
    integer :: i
    integer :: j
    integer :: k
    do i=1,n
      do j=1,n
        c(i,j)=0.
        do k=1,n
          c(i,j)=c(i,j)+a(i,k)*b(k,j)
        end do
      end do
    end do
    return
  end subroutine mtmpld

  ! multiply real nxn matrices
  subroutine mtmpldr(a,b,c,n)
    implicit none
    integer :: n
    real(8) :: a(n,n)
    real(4) :: b(n,n)
    real(4) :: c(n,n)
    integer :: i
    integer :: j
    integer :: k
    do i=1,n
      do j=1,n
        c(i,j)=0.
        do k=1,n
          c(i,j)=c(i,j)+a(i,k)*b(k,j)
        end do
      end do
    end do
    return
  end subroutine mtmpldr


  ! multiply real and int nxn matrices
  subroutine mtmplri(a,b,c,n)
    implicit none
    integer:: n
    real(4) :: a(n,n)
    real(4) :: c(n,n)
    integer :: b(n,n)
    integer :: i
    integer :: j
    integer :: k
    do i=1,n
      do j=1,n
        c(i,j)=0.
        do k=1,n
          c(i,j)=c(i,j)+a(i,k)*b(k,j)
        end do
      end do
    end do
    return
  end subroutine mtmplri

  ! multiply int and real nxn matrices
  subroutine mtmplir(a,b,c,n)
    implicit none
    integer:: n
    integer :: a(n,n)
    real(4) :: b(n,n)
    real(4) :: c(n,n)
    integer :: i
    integer :: j
    integer :: k
    do i=1,n
      do j=1,n
        c(i,j)=0.
        do k=1,n
          c(i,j)=c(i,j)+a(i,k)*b(k,j)
        end do
      end do
    end do
    return
  end subroutine mtmplir


  ! multiply real nxn matrices
  subroutine mtmpli(a,b,c,n)
    implicit none
    integer:: n
    integer :: a(n,n)
    integer :: b(n,n)
    integer :: c(n,n)
    integer :: i
    integer :: j
    integer :: k
    do i=1,n
      do j=1,n
        c(i,j)=0.
        do k=1,n
          c(i,j)=c(i,j)+a(i,k)*b(k,j)
        end do
      end do
    end do
    return
  end subroutine mtmpli

  ! multiply real 6x6 matrices
  subroutine mtmplr3(a,b,c)
    implicit none
    real(4) :: a(3,3)
    real(4) :: b(3,3)
    real(4) :: c(3,3)
    call mtmplr(a,b,c,3)
    return
  end subroutine mtmplr3

  ! multiply real 6x6 matrices
  subroutine mtmpli3(a,b,c)
    implicit none
    integer :: a(3,3)
    integer :: b(3,3)
    integer :: c(3,3)
    call mtmpli(a,b,c,3)
    return
  end subroutine mtmpli3

  ! multiply real 6x6 matrices
  subroutine mtmplr6(a,b,c)
    implicit none
    real(4) :: a(6,6)
    real(4) :: b(6,6)
    real(4) :: c(6,6)
    call mtmplr(a,b,c,6)
    return
  end subroutine mtmplr6

    ! multiply real 6x6 matrices
  subroutine mtmpldr6(a,b,c)
    implicit none
    real(8) :: a(6,6)
    real(4) :: b(6,6)
    real(4) :: c(6,6)
    call mtmpldr(a,b,c,6)
    return
  end subroutine mtmpldr6

  ! multiply integer 6x6 matrices
  subroutine mtmpli6(a,b,c)
    implicit none
    integer :: a(6,6)
    integer :: b(6,6)
    integer :: c(6,6)
    call mtmpli(a,b,c,6)
    return
  end subroutine mtmpli6

  ! multiply real and integer 6x6 matri
  subroutine mtmplri6(a,b,c)
    implicit none
    real(4) :: a(6,6)
    real(4) :: c(6,6)
    integer :: b(6,6)
    call mtmplri(a,b,c,6)
    return
  end subroutine mtmplri6

  ! multiply integer and real 6x6 matri
  subroutine mtmplir6(a,b,c)
    implicit none
    integer :: a(6,6)
    real(4) :: b(6,6)
    real(4) :: c(6,6)
    call mtmplir(a,b,c,6)
    return
  end subroutine mtmplir6

  ! multiply real mxn and mxo matr
  subroutine mtmplmnor(a,b,c,m,n,o)
    implicit none
    integer :: m
    integer :: n
    integer :: o
    real(4) :: a(m,n)
    real(4) :: b(n,o)
    real(4) :: c(m,o)
    integer :: i
    integer :: j
    integer :: k
    do i=1,m
      do j=1,o
        c(i,j)=0.
        do k=1,n
          c(i,j)=c(i,j)+a(i,k)*b(k,j)
        end do
      end do
    end do
    return
  end subroutine mtmplmnor

    ! multiply real mxn and mxo matr
  subroutine mtmplmnod(a,b,c,m,n,o)
    implicit none
    integer :: m
    integer :: n
    integer :: o
    real(8) :: a(m,n)
    real(8) :: b(n,o)
    real(8) :: c(m,o)
    integer :: i
    integer :: j
    integer :: k
    do i=1,m
      do j=1,o
        c(i,j)=0.
        do k=1,n
          c(i,j)=c(i,j)+a(i,k)*b(k,j)
        end do
      end do
    end do
    return
  end subroutine mtmplmnod

  ! multiply integer mxn and mxo m
  subroutine mtmplmnoi(a,b,c,m,n,o)
    implicit none
    integer :: m
    integer :: n
    integer :: o
    integer :: a(m,n)
    integer :: b(n,o)
    integer :: c(m,o)
    integer :: i
    integer :: j
    integer :: k
    do i=1,m
      do j=1,o
        c(i,j)=0.
        do k=1,n
          c(i,j)=c(i,j)+a(i,k)*b(k,j)
        end do
      end do
    end do
    return
  end subroutine mtmplmnoi

  ! multiply integer mxn and real
  subroutine mtmplmnoir(a,b,c,m,n,o)
    implicit none
    integer:: i
    integer:: j
    integer:: k
    integer :: m
    integer :: n
    integer :: o
    integer :: a(m,n)
    real(4) :: b(n,o)
    real(4) :: c(m,o)
    do i=1,m
      do j=1,o
        c(i,j)=0.
        do k=1,n
          c(i,j)=c(i,j)+a(i,k)*b(k,j)
        end do
      end do
    end do
    return
  end subroutine mtmplmnoir

    ! multiply integer mxn and real
  subroutine mtmplmnoid(a,b,c,m,n,o)
    implicit none
    integer:: i
    integer:: j
    integer:: k
    integer :: m
    integer :: n
    integer :: o
    integer :: a(m,n)
    real(8) :: b(n,o)
    real(8) :: c(m,o)
    do i=1,m
      do j=1,o
        c(i,j)=0.
        do k=1,n
          c(i,j)=c(i,j)+a(i,k)*b(k,j)
        end do
      end do
    end do
    return
  end subroutine mtmplmnoid


  ! multiply real mxn and integer
  subroutine mtmplmnori(a,b,c,m,n,o)
    implicit none
    integer :: m
    integer :: n
    integer :: o
    real(4) :: a(m,n)
    real(4) :: c(m,o)
    integer :: b(n,o)
    integer :: i
    integer :: j
    integer :: k
    do i=1,m
      do j=1,o
        c(i,j)=0.
        do k=1,n
          c(i,j)=c(i,j)+a(i,k)*b(k,j)
        end do
      end do
    end do
    return
  end subroutine mtmplmnori

  ! multiply real mxn and integer
  subroutine mtmplmnodi(a,b,c,m,n,o)
    implicit none
    integer :: m
    integer :: n
    integer :: o
    real(8) :: a(m,n)
    real(8) :: c(m,o)
    integer :: b(n,o)
    integer :: i
    integer :: j
    integer :: k
    do i=1,m
      do j=1,o
        c(i,j)=0.
        do k=1,n
          c(i,j)=c(i,j)+a(i,k)*b(k,j)
        end do
      end do
    end do
    return
  end subroutine mtmplmnodi

  ! multiply real mxn and mxo matr
  subroutine mtmplmnor1n(a,b,c,m,n,o)
    implicit none
    integer :: m
    integer :: n
    integer :: o
    real(4) :: a(n)
    real(4) :: b(n,o)
    real(4) :: c(o)
    integer :: i
    integer :: j
    integer :: k
    do i=1,o
      c(i)=0.
      do k=1,n
        c(i)=c(i)+a(k)*b(k,i)
      end do
    end do
    return
  end subroutine mtmplmnor1n

  ! multiply integer mxn and mxo m
  subroutine mtmplmnoi1n(a,b,c,m,n,o)
    implicit none
    integer :: m
    integer :: n
    integer :: o
    integer :: a(n)
    integer :: b(n,o)
    integer :: c(o)
    integer :: i
    integer :: j
    integer :: k
    do i=1,o
      c(i)=0.
      do k=1,n
        c(i)=c(i)+a(k)*b(k,i)
      end do
    end do
    return
  end subroutine mtmplmnoi1n

  ! multiply integer mxn and real
  subroutine mtmplmnoir1n(a,b,c,m,n,o)
    implicit none
    integer:: j
    integer:: k
    integer :: m
    integer :: n
    integer :: o
    integer :: a(n)
    real(4) :: b(n,o)
    real(4) :: c(o)
    do j=1,o
      c(j)=0.
      do k=1,n
        c(j)=c(j)+a(k)*b(k,j)
      end do
    end do
    return
  end subroutine mtmplmnoir1n

  ! multiply real mxn and integer
  subroutine mtmplmnori1n(a,b,c,m,n,o)
    implicit none
    integer :: m
    integer :: n
    integer :: o
    real(4) :: a(n)
    real(4) :: c(o)
    integer :: b(n,o)
    integer :: i
    integer :: j
    integer :: k
    do j=1,o
      c(j)=0.
      do k=1,n
        c(j)=c(j)+a(k)*b(k,j)
      end do
    end do
    return
  end subroutine mtmplmnori1n

  ! multiply real mxn and mxo matr
  subroutine mtmplmnorn1(a,b,c,m,n,o)
    implicit none
    integer :: m
    integer :: n
    integer :: o
    real(4) :: a(m,n)
    real(4) :: b(n)
    real(4) :: c(m)
    integer :: i
    integer :: j
    integer :: k
    do i=1,m
      c(i)=0.
      do k=1,n
        c(i)=c(i)+a(i,k)*b(k)
      end do
    end do
    return
  end subroutine mtmplmnorn1

  ! multiply integer mxn and mxo m
  subroutine mtmplmnoin1(a,b,c,m,n,o)
    implicit none
    integer :: m
    integer :: n
    integer :: o
    integer :: a(m,n)
    integer :: b(n)
    integer :: c(m)
    integer :: i
    integer :: j
    integer :: k
    do i=1,m
      c(i)=0.
      do k=1,n
        c(i)=c(i)+a(i,k)*b(k)
      end do
    end do
    return
  end subroutine mtmplmnoin1

  ! multiply integer mxn and real
  subroutine mtmplmnoirn1(a,b,c,m,n,o)
    implicit none
    integer:: i
    integer:: k
    integer :: m
    integer :: n
    integer :: o
    integer :: a(m,n)
    real(4) :: b(n)
    real(4) :: c(m)
    do i=1,m
      c(i)=0.
      do k=1,n
        c(i)=c(i)+a(i,k)*b(k)
      end do
    end do
    return
  end subroutine mtmplmnoirn1

  ! multiply real mxn and integer
  subroutine mtmplmnorin1(a,b,c,m,n,o)
    implicit none
    integer :: m
    integer :: n
    integer :: o
    real(4) :: a(m,n)
    real(4) :: c(m)
    integer :: b(n)
    integer :: i
    integer :: j
    integer :: k
    do i=1,m
      c(i)=0.
      do k=1,n
        c(i)=c(i)+a(i,k)*b(k)
      end do
    end do
    return
  end subroutine mtmplmnorin1

  ! extract rectangular matrix qm from square matrix q
  subroutine extrrrctmtrx(qm,q,m1,m2,n1,n2)
    real(4),dimension(m2-m1+1,n2-n1+1),intent(inout) :: qm     ! rectangular matrix
    real(4),dimension(n2,n2),intent(inout) :: q         ! square matrix
    integer :: m1                     ! range of row and column
    integer :: m2                     ! range of row and column
    integer :: n1                     ! range of row and column
    integer :: n2                     ! range of row and column
    integer :: i
    integer :: j
    ! qm should be m2-m1+1 x n2-n1+1 matrix
    do i=m1,m2
      do j=n1,n2
        qm(i-m1+1,j-n1+1)=q(i,j)
      end do

       !do j=n1,n2   ! for test
       !   write(6,'(2i4,$)') q%m(i,j)%n1,q%m(i,j)%n2
       !end do
       !write(6,'(i4)') q%m(1,1)%n3  ! for test

       !do j=n1,n2   ! for test
       !   write(6,'(2i4,$)') qm%rm(i-m1+1,j-n1+1)%n1,qm%rm(i-m1+1,j-n1+1)%n2
       !end do
       !write(6,'(i4)') qm%rm(1,1)%n3  ! for test
    end do
  end subroutine extrrrctmtrx

  subroutine wtrrctmtrx(str,q,m,n,frmt)
    character(*) :: str
    character(*) :: frmt
    real(4), dimension(m,n) :: q
    integer :: m
    integer :: n
    integer :: i
    integer :: j
    integer :: n1
    integer :: n2
    integer :: n3
    write(6,'(a)') str
    do i=1,m
      write(6,frmt) (q(i,j),j=1,n)
    end do
    write(6,*)
  end subroutine wtrrctmtrx

  subroutine trnspr(sm,st,n)
    implicit none
    real(4) :: sm(n,n)
    real(4) :: st(n,n)
    integer :: n
    integer :: i
    integer :: j
    do i=1,n
      do j=1,n
        st(i,j)=sm(j,i)
      end do
    end do
    return
  end subroutine trnspr

  subroutine trnspd(sm,st,n)
    implicit none
    real(8) :: sm(n,n)
    real(8) :: st(n,n)
    integer :: n
    integer :: i
    integer :: j
    do i=1,n
      do j=1,n
        st(i,j)=sm(j,i)
      end do
    end do
    return
  end subroutine trnspd


  subroutine trnspi(sm,st,n)
    implicit none
    integer :: sm(n,n)
    integer :: st(n,n)
    integer :: n
    integer :: i
    integer :: j
    do i=1,n
      do j=1,n
        st(i,j)=sm(j,i)
      end do
    end do
    return
  end subroutine trnspi

  subroutine trnspr3(sm)
    implicit none
    real(4) :: smd(3,3)
    real(4) :: sm(3,3)
    !      do i=1,3
    !         do j=1,3
    !            smd(i,j)=sm(j,i)
    !         end do
    !      end do
    call trnspr(sm,smd,3)
    call mcpyr(smd,sm,3)
    return
  end subroutine trnspr3

  subroutine trnspd3(sm)
    implicit none
    real(8) :: smd(3,3)
    real(8) :: sm(3,3)
    !      do i=1,3
    !         do j=1,3
    !            smd(i,j)=sm(j,i)
    !         end do
    !      end do
    call trnspd(sm,smd,3)
    call mcpyd(smd,sm,3)
    return
  end subroutine trnspd3

  subroutine trnspi3(sm)
    implicit none
    integer :: smd(3,3)
    integer :: sm(3,3)
    !      do i=1,3
    !         do j=1,3
    !            smd(i,j)=sm(j,i)
    !         end do
    !      end do
    call trnspi(sm,smd,3)
    call mcpyi(smd,sm,3)
    return
  end subroutine trnspi3

  subroutine trnspr6(sm)
    implicit none
    real(4) :: smd(6,6)
    real(4) :: sm(6,6)
    integer :: i
    integer :: j
    do i=1,6
      do j=1,6
        smd(i,j)=sm(j,i)
      end do
    end do
    call mcpyr(smd,sm,6)
    return
  end subroutine trnspr6

  subroutine trnspd6(sm)
    implicit none
    real(8) :: smd(6,6)
    real(8) :: sm(6,6)
    integer :: i
    integer :: j
    do i=1,6
      do j=1,6
        smd(i,j)=sm(j,i)
      end do
    end do
    call mcpyd(smd,sm,6)
    return
  end subroutine trnspd6

  subroutine trnspi6(sm)
    implicit none
    integer :: smd(6,6)
    integer :: sm(6,6)
    integer :: i
    integer :: j
    do i=1,6
      do j=1,6
        smd(i,j)=sm(j,i)
      end do
    end do
    call mcpyi(smd,sm,6)
    return
  end subroutine trnspi6

  subroutine qsortr1(x,ip,nx)
    real(4) :: x(nx)
    integer :: ip(nx)
    integer :: nx
    integer,allocatable :: st(:,:)
    allocate(st(nx,0:1))
    call qsortr(x,st,ip,nx)
  end subroutine qsortr1

  subroutine qsorti1(x,ip,nx)
    integer :: x(nx)
    integer :: ip(nx)
    integer :: nx
    integer,allocatable :: st(:,:)
    allocate(st(nx,0:1))
    call qsorti(x,st,ip,nx)
  end subroutine qsorti1

  subroutine qsortr(x,st,ip,nx)
    !     quick sort (ascending order of x)
    !     nx: the number of data x
    !     ip: the initial order
    !     st: a work array
    real :: x(nx)
    integer :: st(nx,0:1)
    integer :: ip(nx)
    integer :: nx
    integer :: r,s
    real :: temp,xt
    integer :: itemp
    integer :: i,j,l
    !integer,allocatable :: st(:,:)
    !allocate(st(nx,0:1))

    if(nx==0) return

    do i=1,nx
      ip(i)=i
    end do

    s=1
    st(1,0)=1
    st(1,1)=nx

    !1 continue
    call setlrs()
    !2 continue
    call setijxt()
    do
      !3 continue
      do
        if(i<nx) then
          if(x(i)<xt) then
            i=i+1
            cycle
          else
            exit
          endif
        else
          exit
        endif
      end do

      do
        if(j>1) then
          if(xt<x(j)) then
            j=j-1
            cycle
          else
            exit
          endif
        else
          exit
        endif
      end do

      if(i<=j) then
        temp=x(j)
        x(j)=x(i)
        x(i)=temp
        itemp=ip(j)
        ip(j)=ip(i)
        ip(i)=itemp
        if(i<=nx.and.j>=1) then
          i=i+1
          j=j-1
          cycle
        else
          exit
        end if
      endif

      if(j-l>r-i) then
        if(l<j) then
          s=s+1
          st(s,0)=l
          st(s,1)=j
        endif
        l=i
      else
        if(i<r) then
          s=s+1
          st(s,0)=i
          st(s,1)=r
        endif
        r=j
      end if

      if(l<r) then
        call setijxt()
        cycle
      end if
      if(s/=0) then
        call setlrs()
        call setijxt()
        cycle
      else
        exit
      end if
    end do
    return

  contains

    subroutine setlrs()
      l=st(s,0)
      r=st(s,1)
      s=s-1
    end subroutine

    subroutine setijxt()
      i=l
      j=r
      xt=x((l+r)/2)
    end subroutine
  end subroutine

  subroutine qsorti(x,st,ip,nx)
    !     quick sort (ascending order of x)
    !     nx: the number of data x
    !     ip: the initial order
    !     st: a work array
    integer :: x(nx)
    integer :: st(nx,0:1)
    integer :: ip(nx)
    integer :: nx
    integer :: r,s
    integer :: temp,xt,itemp
    integer :: i,j,l
    !integer,allocatable :: st(:,:)
    !allocate(st(nx,0:1))

    if(nx==0) return

    do i=1,nx
      ip(i)=i
    end do

    s=1
    st(1,0)=1
    st(1,1)=nx

    !1 continue
    call setlrs()
    !2 continue
    call setijxt()
    do
      !3 continue
      do
        if(i<nx) then
          if(x(i)<xt) then
            i=i+1
            cycle
          else
            exit
          endif
        else
          exit
        endif
      end do

      do
        if(j>1) then
          if(xt<x(j)) then
            j=j-1
            cycle
          else
            exit
          endif
        else
          exit
        endif
      end do

      if(i<=j) then
        temp=x(j)
        x(j)=x(i)
        x(i)=temp
        itemp=ip(j)
        ip(j)=ip(i)
        ip(i)=itemp
        if(i<=nx.and.j>=1) then
          i=i+1
          j=j-1
          cycle
        else
          exit
        end if
      endif

      if(j-l>r-i) then
        if(l<j) then
          s=s+1
          st(s,0)=l
          st(s,1)=j
        endif
        l=i
      else
        if(i<r) then
          s=s+1
          st(s,0)=i
          st(s,1)=r
        endif
        r=j
      end if

      if(l<r) then
        call setijxt()
        cycle
      end if
      if(s/=0) then
        call setlrs()
        call setijxt()
        cycle
      else
        exit
      end if
    end do
    return

  contains

    subroutine setlrs()
      l=st(s,0)
      r=st(s,1)
      s=s-1
    end subroutine

    subroutine setijxt()
      i=l
      j=r
      xt=x((l+r)/2)
    end subroutine
  end subroutine


  !  subroutine qsortr(x,st,ip,nx)
  !    implicit none
  !    !     quick sort (ascending order of x)
  !    !     nx: the number of data x
  !    !     ip: the initial order
  !    !     st: a work array
  !    integer :: nx
  !    real(4) :: x(nx)
  !    integer :: st(nx,0:1)
  !    integer :: ip(nx)
  !    real(4) :: xt
  !    real(4) ::temp
  !    integer :: r
  !    integer :: s
  !    integer :: i
  !    integer :: j
  !    integer :: l
  !    integer :: itemp
  !
  !    if(nx<=0) return
  !    do i=1,nx
  !      ip(i)=i
  !    end do
  !    s=1
  !    st(1,0)=1
  !    st(1,1)=nx
  !1   l=st(s,0)
  !    r=st(s,1)
  !    s=s-1
  !2   i=l
  !    j=r
  !    xt=x((l+r)/2)
  !3   if(i<nx) then
  !      if(x(i)<xt) then
  !        i=i+1
  !        go to 3
  !      endif
  !    endif
  !4   if(j>1) then
  !      if(xt<x(j)) then
  !        j=j-1
  !        go to 4
  !      endif
  !    endif
  !    if(i<=j) then
  !      temp=x(j)
  !      x(j)=x(i)
  !      x(i)=temp
  !      itemp=ip(j)
  !      ip(j)=ip(i)
  !      ip(i)=itemp
  !      if(i<=nx.and.j>=1) then
  !        i=i+1
  !        j=j-1
  !        go to 3
  !      end if
  !    endif
  !    if(j-l<=r-i) go to 5
  !    if(i<r) then
  !      s=s+1
  !      st(s,0)=i
  !      st(s,1)=r
  !    endif
  !    r=j
  !    go to 6
  !5   if(l<j) then
  !      s=s+1
  !      st(s,0)=l
  !      st(s,1)=j
  !    endif
  !    l=i
  !6   if(l<r) go to 2
  !    if(s/=0) go to 1
  !    return
  !  end subroutine qsortr
  !
  !  subroutine qsorti(x,st,ip,nx)
  !    implicit none
  !    !     quick sort (ascending order of x)
  !    !     nx: the number of data x
  !    !     ip: the initial order
  !    !     st: a work array
  !    integer :: nx
  !    integer :: x(nx)
  !    integer :: st(nx,0:1)
  !    integer :: ip(nx)
  !    real(4) :: xt
  !    real(4) ::temp
  !    integer :: r
  !    integer :: s
  !    integer :: i
  !    integer :: j
  !    integer :: l
  !    integer :: itemp
  !
  !    if(nx<=0) return
  !    do i=1,nx
  !      ip(i)=i
  !    end do
  !    s=1
  !    st(1,0)=1
  !    st(1,1)=nx
  !1   l=st(s,0)
  !    r=st(s,1)
  !    s=s-1
  !2   i=l
  !    j=r
  !    xt=x((l+r)/2)
  !3   if(i<nx) then
  !      if(x(i)<xt) then
  !        i=i+1
  !        go to 3
  !      endif
  !    endif
  !4   if(j>1) then
  !      if(xt<x(j)) then
  !        j=j-1
  !        go to 4
  !      endif
  !    endif
  !    if(i<=j) then
  !      temp=x(j)
  !      x(j)=x(i)
  !      x(i)=temp
  !      itemp=ip(j)
  !      ip(j)=ip(i)
  !      ip(i)=itemp
  !      if(i<=nx.and.j>=1) then
  !        i=i+1
  !        j=j-1
  !        go to 3
  !      end if
  !    endif
  !    if(j-l<=r-i) go to 5
  !    if(i<r) then
  !      s=s+1
  !      st(s,0)=i
  !      st(s,1)=r
  !    endif
  !    r=j
  !    go to 6
  !5   if(l<j) then
  !      s=s+1
  !      st(s,0)=l
  !      st(s,1)=j
  !    endif
  !    l=i
  !6   if(l<r) go to 2
  !    if(s/=0) go to 1
  !    return
  !  end subroutine qsorti

  subroutine mttranspr(a,b,n)  ! transpose matrix rectangular matrix
    implicit none
    real(4), dimension(n,n) :: a
    real(4), dimension(n,n) ::b
    integer,intent(in) :: n
    integer :: i
    integer :: j
    do i=1,n
      do j=1,n
        b(j,i) = a(i,j)
      end do
    end do
    return
  end subroutine mttranspr

  subroutine mttranspi(a,b,n)  ! transpose matrix rectangular matrix
    implicit none
    integer,dimension(n,n) :: a
    integer,dimension(n,n) ::b
    integer,intent(in) :: n
    integer :: i
    integer :: j
    do i=1,n
      do j=1,n
        b(j,i) = a(i,j)
      end do
    end do
    return
  end subroutine mttranspi

  subroutine rctmttranspr(a,b,m,n)  ! transpose matrix rectangular matrix
    implicit none
    real(4), dimension(m,n) :: a
    real(4), dimension(n,m) ::b
    integer,intent(in) :: m
    integer,intent(in) ::n
    integer :: i
    integer :: j
    do i=1,m
      do j=1,n
        b(j,i) = a(i,j)
      end do
    end do
    return
  end subroutine rctmttranspr

  subroutine mcpyr(s1,s2,n)
    implicit none
    real(4) :: s1(n,n)
    real(4) :: s2(n,n)
    integer :: n
    integer :: i
    integer :: j
    do i=1,n
      do j=1,n
        s2(i,j)=s1(i,j)
      end do
    end do
    return
  end subroutine mcpyr

  subroutine mcpyd(s1,s2,n)
    implicit none
    real(8) :: s1(n,n)
    real(8) :: s2(n,n)
    integer :: n
    integer :: i
    integer :: j
    do i=1,n
      do j=1,n
        s2(i,j)=s1(i,j)
      end do
    end do
    return
  end subroutine mcpyd

  subroutine mcpyi(s1,s2,n)
    implicit none
    integer :: s1(n,n)
    integer :: s2(n,n)
    integer :: n
    integer :: i
    integer :: j
    do i=1,n
      do j=1,n
        s2(i,j)=s1(i,j)
      end do
    end do
    return
  end subroutine mcpyi

  subroutine mcpyir(s1,s2,n)
    implicit none
    integer :: s1(n,n)
    real(4) :: s2(n,n)
    integer :: n
    integer :: i
    integer :: j
    do i=1,n
      do j=1,n
        s2(i,j)=s1(i,j)
      end do
    end do
    return
  end subroutine mcpyir

  subroutine mcpyid(s1,s2,n)
    implicit none
    integer :: s1(n,n)
    real(8) :: s2(n,n)
    integer :: n
    integer :: i
    integer :: j
    do i=1,n
      do j=1,n
        s2(i,j)=s1(i,j)
      end do
    end do
    return
  end subroutine mcpyid

  subroutine mcpyr3(s1,s2)
    implicit none
    real(4) :: s1(3,3)
    real(4) :: s2(3,3)
    integer :: i
    integer :: j
    do i=1,3
      do j=1,3
        s2(i,j)=s1(i,j)
      end do
    end do
    return
  end subroutine mcpyr3

  subroutine mcpyd3(s1,s2)
    implicit none
    real(8) :: s1(3,3)
    real(8) :: s2(3,3)
    integer :: i
    integer :: j
    do i=1,3
      do j=1,3
        s2(i,j)=s1(i,j)
      end do
    end do
    return
  end subroutine mcpyd3

  subroutine mcpydr3(s1,s2)
    implicit none
    real(8) :: s1(3,3)
    real(4) :: s2(3,3)
    integer :: i
    integer :: j
    do i=1,3
      do j=1,3
        s2(i,j)=s1(i,j)
      end do
    end do
    return
  end subroutine mcpydr3

  subroutine mcpyi3(s1,s2)
    implicit none
    integer :: s1(3,3)
    integer :: s2(3,3)
    integer :: i
    integer :: j
    do i=1,3
      do j=1,3
        s2(i,j)=s1(i,j)
      end do
    end do
    return
  end subroutine mcpyi3

  subroutine mtxveci(r,x,rx,n)
    integer :: r(n,n)
    integer :: x(n)
    integer :: rx(n)
    integer :: n
    integer :: i
    integer :: j
    do i=1,n
      rx(i)=0
      do j=1,n
        rx(i)=rx(i)+r(i,j)*x(j)
      end do
    end do
  end subroutine mtxveci

  subroutine mtxvecr(r,x,rx,n)
    real(4) :: r(n,n)
    real(4) :: x(n)
    real(4) :: rx(n)
    integer :: n
    integer :: i
    integer :: j
    do i=1,n
      rx(i)=0
      do j=1,n
        rx(i)=rx(i)+r(i,j)*x(j)
      end do
    end do
  end subroutine mtxvecr

  subroutine mtxvecd(r,x,rx,n)
    real(8) :: r(n,n)
    real(8) :: x(n)
    real(8) :: rx(n)
    integer :: n
    integer :: i
    integer :: j
    do i=1,n
      rx(i)=0
      do j=1,n
        rx(i)=rx(i)+r(i,j)*x(j)
      end do
    end do
  end subroutine mtxvecd

  subroutine mtxvecir(r,x,rx,n)
    integer :: r(n,n)
    real(4) :: x(n)
    real(4) :: rx(n)
    integer :: n
    integer :: i
    integer :: j
    do i=1,n
      rx(i)=0
      do j=1,n
        rx(i)=rx(i)+r(i,j)*x(j)
      end do
    end do
  end subroutine mtxvecir

  subroutine mtxvecid(r,x,rx,n)
    integer :: r(n,n)
    real(8) :: x(n)
    real(8) :: rx(n)
    integer :: n
    integer :: i
    integer :: j
    do i=1,n
      rx(i)=0
      do j=1,n
        rx(i)=rx(i)+r(i,j)*x(j)
      end do
    end do
  end subroutine mtxvecid

  subroutine mtxvecrd(r,x,rx,n)
    real(4) :: r(n,n)
    real(8) :: x(n)
    real(8) :: rx(n)
    integer :: n
    integer :: i
    integer :: j
    do i=1,n
      rx(i)=0
      do j=1,n
        rx(i)=rx(i)+r(i,j)*x(j)
      end do
    end do
  end subroutine mtxvecrd

  subroutine mtxvecdr(r,x,rx,n)
    real(8) :: r(n,n)
    real(4) :: x(n)
    real(4) :: rx(n)
    integer :: n
    integer :: i
    integer :: j
    do i=1,n
      rx(i)=0
      do j=1,n
        rx(i)=rx(i)+r(i,j)*x(j)
      end do
    end do
  end subroutine mtxvecdr


  subroutine vecxmti(x,r,xr,n)
    integer :: x(n)
    integer :: r(n,n)
    integer :: xr(n)
    integer :: n
    integer :: i
    integer :: j
    do i=1,n
      xr(i)=0
      do j=1,n
        xr(i)=xr(i)+x(j)*r(j,i)
      end do
    end do
  end subroutine vecxmti

  subroutine vecxmtr(x,r,xr,n)
    real(4) :: x(n)
    real(4) :: r(n,n)
    real(4) :: xr(n)
    integer :: n
    integer :: i
    integer :: j
    do i=1,n
      xr(i)=0
      do j=1,n
        xr(i)=xr(i)+x(j)*r(j,i)
      end do
    end do
  end subroutine vecxmtr

  subroutine vecxmtd(x,r,xr,n)
    real(8) :: x(n)
    real(8) :: r(n,n)
    real(8) :: xr(n)
    integer :: n
    integer :: i
    integer :: j
    do i=1,n
      xr(i)=0
      do j=1,n
        xr(i)=xr(i)+x(j)*r(j,i)
      end do
    end do
  end subroutine vecxmtd

  subroutine vecxmtrd(x,r,xr,n)
    real(4) :: x(n)
    real(8) :: r(n,n)
    real(8) :: xr(n)
    integer :: n
    integer :: i
    integer :: j
    do i=1,n
      xr(i)=0
      do j=1,n
        xr(i)=xr(i)+x(j)*r(j,i)
      end do
    end do
  end subroutine vecxmtrd

  subroutine vecxmtdr(x,r,xr,n)
    real(8) :: x(n)
    real(4) :: r(n,n)
    real(4) :: xr(n)
    integer :: n
    integer :: i
    integer :: j
    do i=1,n
      xr(i)=0
      do j=1,n
        xr(i)=xr(i)+x(j)*r(j,i)
      end do
    end do
  end subroutine vecxmtdr


  subroutine vecxmtri(x,r,xr,n)
    real(4) :: x(n)
    integer :: r(n,n)
    real(4) :: xr(n)
    integer :: n
    integer :: i
    integer :: j
    do i=1,n
      xr(i)=0
      do j=1,n
        xr(i)=xr(i)+x(j)*r(j,i)
      end do
    end do
  end subroutine vecxmtri

  subroutine vecxmtdi(x,r,xr,n)
    real(8) :: x(n)
    integer :: r(n,n)
    real(8) :: xr(n)
    integer :: n
    integer :: i
    integer :: j
    do i=1,n
      xr(i)=0
      do j=1,n
        xr(i)=xr(i)+x(j)*r(j,i)
      end do
    end do
  end subroutine vecxmtdi

  subroutine vecxmtir(x,r,xr,n)
    integer :: x(n)
    real(4) :: r(n,n)
    real(4) :: xr(n)
    integer :: n
    integer :: i
    integer :: j
    do i=1,n
      xr(i)=0
      do j=1,n
        xr(i)=xr(i)+x(j)*r(j,i)
      end do
    end do
  end subroutine vecxmtir

  subroutine vecxmtid(x,r,xr,n)
    integer :: x(n)
    real(8) :: r(n,n)
    real(8) :: xr(n)
    integer :: n
    integer :: i
    integer :: j
    do i=1,n
      xr(i)=0
      do j=1,n
        xr(i)=xr(i)+x(j)*r(j,i)
      end do
    end do
  end subroutine vecxmtid


  !  subroutine mtxvecr1(r,x,rx,n)
  !    integer :: r(n,n)
  !    real(4) :: x(n)
  !    real(4) :: rx(n)
  !    integer :: n
  !    integer :: i
  !    integer :: j
  !    do i=1,n
  !      rx(i)=0
  !      do j=1,n
  !        rx(i)=rx(i)+r(i,j)*x(j)
  !      end do
  !    end do
  !  end subroutine mtxvecr1

  ! zero vectors
  !subroutine zerov(v1,n)
  !  real(4) v1(:)
  !  integer n,j
  !  do j=1,n
  !     v1(j)=0
  !  end do
  !  return
  !end subroutine zerov

  ! zero vectors
  subroutine zerovr(v1,n)
    implicit none
    real(4) :: v1(:)
    integer :: n
    integer :: j
    do j=1,n
      v1(j)=0
    end do
    return
  end subroutine zerovr

    ! zero vectors
  subroutine zerovd(v1,n)
    implicit none
    real(8) :: v1(:)
    integer :: n
    integer :: j
    do j=1,n
      v1(j)=0
    end do
    return
  end subroutine zerovd

  ! complex zero vectors
  subroutine zerovc(v1,n)
    implicit none
    complex :: v1(:)
    integer :: n
    integer :: j
    do j=1,n
      v1(j)=0
    end do
    return
  end subroutine zerovc

  ! zero integer vectors
  subroutine zerovi(v1,n)
    implicit none
    integer :: v1(:)
    integer :: n
    integer :: j
    do j=1,n
      v1(j)=0
    end do
    return
  end subroutine zerovi

  ! zero vectors
  integer function ifzerovr(v1,n)
    implicit none
    real(4) :: v1(n)
    integer :: n
    integer :: j
    do j=1,n
      if(v1(j)/=0) then
        ifzerovr=0
        return
      end if
    end do
    ifzerovr=1
    return
  end function ifzerovr

  ! zero vectors
  integer function ifzerovi(v1,n)
    implicit none
    integer :: v1(n)
    integer :: n
    integer :: j
    do j=1,n
      if(v1(j)/=0) then
        ifzerovi=0
        return
      end if
    end do
    ifzerovi=1
    return
  end function ifzerovi

  ! multiplied by fact
  subroutine vfact(v1,fact,n)
    implicit none
    real(4) :: fact
    real(4) :: v1(n)
    integer :: n
    integer :: j
    do j=1,n
      v1(j)=v1(j)*fact
    end do
    return
  end subroutine vfact

  subroutine vmult(v1,v2,v3,n)
    implicit none
    real(4) :: v1(n)
    real(4) :: v2(n)
    real(4) :: v3(n)
    integer :: n
    integer :: j
    do j=1,n
      v3(j)=v1(j)*v2(j)
    end do
    return
  end subroutine vmult

  subroutine vsclr(v1,scl,v2,n)
    implicit none
    real(4) :: scl
    real(4) :: v1(n)
    real(4) :: v2(n)
    integer :: n
    integer :: i
    do i=1,n
      v2(i)=v1(i)*scl
    end do
  end subroutine vsclr

  ! scale real vector
  subroutine sclvecr(vctr,scl,n)
    implicit none
    real(4) :: vctr(n)
    real(4) :: scl
    integer :: n
    integer :: i
    do i=1,n
      vctr(i)=vctr(i)*scl
    end do
    return
  end subroutine sclvecr

  ! copy i-th row in nxn matr
  subroutine getrowmtrxr(mtrx,vctr,n,i)
    implicit none
    real(4) :: mtrx(n,n)
    real(4) :: vctr(n)
    integer :: n
    integer :: i
    integer :: j
    do j=1,n
      vctr(j)=mtrx(i,j)
    end do
    return
  end subroutine getrowmtrxr

  ! copy i-th row in nxn matr
  subroutine getrowmtrxi(mtrx,vctr,n,i)
    implicit none
    integer :: mtrx(n,n)
    integer :: vctr(n)
    integer :: n
    integer :: i
    integer :: j
    do j=1,n
      vctr(j)=mtrx(i,j)
    end do
    return
  end subroutine getrowmtrxi

  ! copy vctr into i-th row o
  subroutine setrowmtrxr(vctr,mtrx,n,i)
    implicit none
    real(4) :: mtrx(n,n)
    real(4) :: vctr(n)
    integer :: n
    integer :: i
    integer :: j
    do j=1,n
      mtrx(i,j)=vctr(j)
    end do
    return
  end subroutine setrowmtrxr

  ! copy vctr into i-th row o
  subroutine setrowmtrxi(vctr,mtrx,n,i)
    implicit none
    integer :: mtrx(n,n)
    integer :: vctr(n)
    integer :: n
    integer :: i
    integer :: j
    do j=1,n
      mtrx(i,j)=vctr(j)
    end do
    return
  end subroutine setrowmtrxi

  ! scale real matrix
  subroutine sclmtrxr(mtrx,scl,n)
    implicit none
    real(4) :: mtrx(n,n)
    real(4) :: scl
    integer :: n
    integer :: i
    integer :: j
    do i=1,n
      do j=1,n
        mtrx(i,j)=mtrx(i,j)*scl
      end do
    end do
    return
  end subroutine sclmtrxr

  ! scale real matrix
  subroutine sclmtrxdr(mtrx,scl,n)
    implicit none
    real(8) :: mtrx(n,n)
    real(4):: scl
    integer :: n
    integer :: i
    integer :: j
    do i=1,n
      do j=1,n
        mtrx(i,j)=mtrx(i,j)*scl
      end do
    end do
    return
  end subroutine sclmtrxdr

  ! scale real matrix
  subroutine sclmtrxd(mtrx,scl,n)
    implicit none
    real(8) :: mtrx(n,n)
    real(8):: scl
    integer :: n
    integer :: i
    integer :: j
    do i=1,n
      do j=1,n
        mtrx(i,j)=mtrx(i,j)*scl
      end do
    end do
    return
  end subroutine sclmtrxd

  subroutine vcpyr(dd,de,n)
    implicit none
    real(4) :: dd(:)
    real(4) :: de(:)
    integer :: n
    integer :: j
    do j=1,n
      de(j)=dd(j)
    end do
    return
  end subroutine vcpyr

  subroutine vcpyd(dd,de,n)
    implicit none
    real(8) :: dd(:)
    real(8) :: de(:)
    integer :: n
    integer :: j
    do j=1,n
      de(j)=dd(j)
    end do
    return
  end subroutine vcpyd

  subroutine vcpyi(dd,de,n)
    implicit none
    integer :: dd(:)
    integer :: de(:)
    integer :: n
    integer :: j
    do j=1,n
      de(j)=dd(j)
    end do
    return
  end subroutine vcpyi

  ! copy integer vector to real vector
  subroutine vcpyir(dd,de,n)
    implicit none
    integer :: dd(:)
    real(4) :: de(:)
    integer :: n
    integer :: j
    do j=1,n
      de(j)=dd(j)
    end do
    return
  end subroutine vcpyir

  subroutine negater(v1,n)
    implicit none
    real(4) :: v1(:)
    integer :: n
    integer :: j
    do j=1,n
      v1(j)=-v1(j)
    end do
    return
  end subroutine negater

  subroutine negatei(v1,n)
    implicit none
    integer :: v1(:)
    integer :: n
    integer :: j
    do j=1,n
      v1(j)=-v1(j)
    end do
    return
  end subroutine negatei

  subroutine vsumr(x1,x2,x3,n)
    implicit none
    real(4), dimension(:) :: x1
    real(4), dimension(:) ::x2
    real(4), dimension(:) ::x3
    integer :: n
    integer :: j
    do j=1,n
      x3(j)=x1(j)+x2(j)
    end do
    return
  end subroutine vsumr

  subroutine vsumi(x1,x2,x3,n)
    implicit none
    integer,dimension(:) :: x1
    integer,dimension(:) ::x2
    integer,dimension(:) ::x3
    integer :: n
    integer :: j
    do j=1,n
      x3(j)=x1(j)+x2(j)
    end do
    return
  end subroutine vsumi

  subroutine vsumir(x1,x2,x3,n)
    implicit none
    integer,dimension(:) :: x1
    real(4), dimension(:) :: x2
    real(4), dimension(:) ::x3
    integer :: n
    integer :: j
    do j=1,n
      x3(j)=x1(j)+x2(j)
    end do
    return
  end subroutine vsumir

  subroutine vsubr(x1,x2,x3,n)
    implicit none
    real(4), dimension(:) :: x1
    real(4), dimension(:) ::x2
    real(4), dimension(:) ::x3
    integer :: n
    integer :: j
    do j=1,n
      x3(j)=x1(j)-x2(j)
    end do
    return
  end subroutine vsubr

  subroutine vsubi(x1,x2,x3,n)
    implicit none
    integer,dimension(:) :: x1
    integer,dimension(:) ::x2
    integer,dimension(:) ::x3
    integer :: n
    integer :: j
    do j=1,n
      x3(j)=x1(j)-x2(j)
    end do
    return
  end subroutine vsubi

  !  ! vector product of 3d vectors
  !  subroutine vprd(q1,q2,q3)
  !    real(4) q1(:),q2(:),q3(:)
  !    call vprdr(q1,q2,q3)
  !    return
  !  end subroutine vprd

  ! vector product of 3d vectors
  subroutine vprdi(q1,q2,q3)
    implicit none
    integer,dimension(:) :: q1
    integer,dimension(:) ::q2
    integer,dimension(:) ::q3
    q3(1)=q1(2)*q2(3)-q1(3)*q2(2)
    q3(2)=q1(3)*q2(1)-q1(1)*q2(3)
    q3(3)=q1(1)*q2(2)-q1(2)*q2(1)
    return
  end subroutine vprdi

  ! vector product of 3d vectors
  subroutine vprdr(q1,q2,q3)
    implicit none
    real(4), dimension(:) :: q1
    real(4), dimension(:) ::q2
    real(4), dimension(:) ::q3
    q3(1)=q1(2)*q2(3)-q1(3)*q2(2)
    q3(2)=q1(3)*q2(1)-q1(1)*q2(3)
    q3(3)=q1(1)*q2(2)-q1(2)*q2(1)
    return
  end subroutine vprdr

  subroutine chsgnr(q)
    implicit none
    real(4) :: q(:)
    integer :: i
    do i=1,6
      q(i)=-q(i)
    end do
    return
  end subroutine chsgnr

  subroutine chsgnd(q)
    implicit none
    real(8) :: q(:)
    integer :: i
    do i=1,6
      q(i)=-q(i)
    end do
    return
  end subroutine chsgnd


  subroutine vnegater(x,n)
    implicit none
    !*****negate real vector
    real(4) :: x(:)
    integer :: n
    integer :: i
    do i=1,n
      x(i)=-x(i)
    end do
    return
  end subroutine vnegater

  subroutine vnegatei(x,n)
    implicit none
    !*****negate integer vector
    integer :: x(:)
    integer :: n
    integer :: i
    do i=1,n
      x(i)=-x(i)
    end do
    return
  end subroutine vnegatei

  real(4) function vinpr(x,y,n)
    implicit none
    !*****inner product of n-dimensional vactors, x,y
    real(4) :: x(:)
    real(4) :: y(:)
    integer :: n
    integer :: i
    vinpr=0
    do i=1,n
      vinpr=vinpr+x(i)*y(i)
    end do
    return
  end function vinpr

  real(8) function vinpd(x,y,n)
    implicit none
    !*****inner product of n-dimensional vactors, x,y
    real(8) :: x(:)
    real(8) :: y(:)
    integer :: n
    integer :: i
    vinpd=0
    do i=1,n
      vinpd=vinpd+x(i)*y(i)
    end do
    return
  end function vinpd

  integer function vinpi(x,y,n)
    implicit none
    !*****inner product of n-dimensional vactors, x,y
    integer :: x(:)
    integer :: y(:)
    integer :: n
    integer :: i
    vinpi=0
    do i=1,n
      vinpi=vinpi+x(i)*y(i)
    end do
    return
  end function vinpi

  complex(4) function vinpc(x,y,n)
    implicit none
    !*****inner product of n-dimensional vactors, x,y
    complex :: x(:)
    complex :: y(:)
    integer :: n
    integer :: i
    vinpc=0
    do i=1,n
      vinpc=vinpc+conjg(x(i))*y(i)
    end do
    return
  end function vinpc

  complex(8) function vinpz(x,y,n)
    implicit none
    !*****inner product of n-dimensional vactors, x,y
    complex(8) :: x(:)
    complex(8) :: y(:)
    integer :: n
    integer :: i
    vinpz=0
    do i=1,n
      vinpz=vinpz+conjg(x(i))*y(i)
    end do
    return
  end function vinpz

  real(4) function vinpir(x,y,n)
    implicit none
    !*****inner product of n-dimensional vactors, x,y
    integer :: x(:)
    real(4) :: y(:)
    integer :: n
    integer :: i
    vinpir=0
    do i=1,n
      vinpir=vinpir+x(i)*y(i)
    end do
    return
  end function vinpir

  real(4) function vinpri(x,h,n)
    implicit none
    !*****inner product of n-dimensional integral and real vectors
    real(4) :: x(:)
    integer :: h(:)
    integer :: n
    integer :: i
    vinpri=0
    do i=1,n
      vinpri=vinpri+h(i)*x(i)
    end do
    return
  end function vinpri

  real(4) function vabsr(x,n)
    implicit none
    real(4) :: x(:)
    !*****absolute value of n-dimensional vector x
    integer :: n
    integer :: i
    vabsr=0
    do i=1,n
      vabsr=vabsr+x(i)**2
    end do
    vabsr=sqrt(vabsr)
    return
  end function vabsr

  real(8) function vabsd(x,n)
    implicit none
    real(8):: x(:)
    !*****absolute value of n-dimensional vector x
    integer :: n
    integer :: i
    vabsd=0
    do i=1,n
      vabsd=vabsd+x(i)**2
    end do
    vabsd=sqrt(vabsd)
    return
  end function vabsd

  subroutine diagmtrxr(da,a,n)
    implicit none
    real(4) :: da(n)
    real(4) :: a(n,n)
    !     get real diagonal matrix a with diagonal part da
    integer :: n
    integer :: i
    call zeromtrxr(a,n)
    do i=1,n
      a(i,i)=da(i)
    end do
    return
  end subroutine diagmtrxr

  subroutine diagmtrxd(da,a,n)
    implicit none
    real(8) :: da(n)
    real(8) :: a(n,n)
    !     get real diagonal matrix a with diagonal part da
    integer :: n
    integer :: i
    call zeromtrxd(a,n)
    do i=1,n
      a(i,i)=da(i)
    end do
    return
  end subroutine diagmtrxd

  subroutine diagmtrxi(da,a,n)
    implicit none
    integer :: da(n)
    integer :: a(n,n)
    integer :: n
    integer :: i
    !     get integer diagonal matrix a with diagonal part da
    call zeromtrxi(a,n)
    do i=1,n
      a(i,i)=da(i)
    end do
    return
  end subroutine diagmtrxi

  subroutine unitmtrxi(a,n)
    implicit none
    integer :: a(n,n)
    integer :: n
    integer :: i
    integer :: j
    a(:,:)=0
    do i=1,n
      a(i,i)=1
    end do
    return
  end subroutine unitmtrxi

  subroutine unitmtrxr(a,n)
    implicit none
    real(4) :: a(n,n)
    integer :: n
    integer :: i
    integer :: j
    a(:,:)=0
    do i=1,n
      a(i,i)=1.
    end do
    return
  end subroutine unitmtrxr

  subroutine unitmtrxd(a,n)
    implicit none
    real(8) :: a(n,n)
    integer :: n
    integer :: i
    integer :: j
    a(:,:)=0
    do i=1,n
      a(i,i)=1.
    end do
    return
  end subroutine unitmtrxd


  ! zero matrix
  subroutine zeromtrxr(m,n)
    implicit none
    real(4) :: m(n,n)
    integer :: n
    integer :: i
    integer :: j
    m(:,:)=0
    return
  end subroutine zeromtrxr

    ! zero matrix
  subroutine zeromtrxd(m,n)
    implicit none
    real(8) :: m(n,n)
    integer :: n
    integer :: i
    integer :: j
    m(:,:)=0
    return
  end subroutine zeromtrxd


  ! zero matrix
  subroutine zeromtrxi(m,n)
    implicit none
    integer :: m(n,n)
    integer :: n
    integer :: i
    integer :: j
    m(:,:)=0
    return
  end subroutine zeromtrxi

  subroutine normlr(x,n)
    implicit none
    real(4) :: absh
    real(4) :: eps
    real(4) :: x(:)
    integer :: n
    integer :: i
    data eps /0.000001/
    !     normalization of x
    absh=vabs(x,n)
    if(absh>eps) then
      do i=1,n
        x(i)=x(i)/absh
      end do
      return
    end if
    write(6,'(*(g0,1x))') 'cannt normalize x : x=',x(1:n); stop
    return
  end subroutine normlr

  subroutine normld(x,n)
    implicit none
    real(8):: absh
    real(8):: eps
    real(8) :: x(:)
    integer :: n
    integer :: i
    data eps /0.000001/
    !     normalization of x
    absh=vabs(x,n)
    if(absh>eps) then
      do i=1,n
        x(i)=x(i)/absh
      end do
      return
    end if
    write(6,'(*(g0,1x))') 'cannt normalize x : x=',x(1:n); stop
    return
  end subroutine normld

  subroutine wtvctrr(str,v,n)
    implicit none
    character(*) :: str
    integer :: n
    integer :: j
    real(4) :: v(:)
    write(6,'(a,$)') str
    write(6,*) (v(j),j=1,n)
    return
  end subroutine wtvctrr

  subroutine wtvctri(str,v,n)
    implicit none
    character(*) :: str
    integer :: v(:)
    integer :: n
    integer :: j
    write(6,'(a,$)') str
    write(6,*) (v(j),j=1,n)
    return
  end subroutine wtvctri

  subroutine wtvctrrf(str,v,n,frmt)
    implicit none
    character(*) :: str
    character(*) :: frmt
    real(4) :: v(:)
    integer :: n
    integer :: j
    write(6,'(a,$)') str
    write(6,frmt) (v(j),j=1,n)
    return
  end subroutine wtvctrrf

  subroutine wtvctrif(str,v,n,frmt)
    implicit none
    character(*) :: str
    character(*) :: frmt
    integer :: v(:)
    integer :: n
    integer :: j
    write(6,'(a,$)') str
    write(6,frmt) (v(j),j=1,n)
    return
  end subroutine wtvctrif

  subroutine wtmtrxi(str,mtrx,n)
    implicit none
    character(*) :: str
    integer :: mtrx(n,n)
    integer :: n
    integer :: i
    integer :: j
    write(6,'(a)') str
    do i=1,n
      write(6,*) (mtrx(i,j),j=1,n)
    end do
    return
  end subroutine wtmtrxi

  subroutine wtmtrxr(str,mtrx,n)
    implicit none
    character(*) :: str
    real(4) :: mtrx(n,n)
    integer :: n
    integer :: i
    integer :: j
    write(6,'(a)') str
    do i=1,n
      write(6,*) (mtrx(i,j),j=1,n)
    end do
    return
  end subroutine wtmtrxr

  subroutine wtmtrxd(str,mtrx,n)
    implicit none
    character(*) :: str
    real(8) :: mtrx(n,n)
    integer :: n
    integer :: i
    integer :: j
    write(6,'(a)') str
    do i=1,n
      write(6,*) (mtrx(i,j),j=1,n)
    end do
    return
  end subroutine wtmtrxd

  subroutine wtmtrxif(str,mtrx,n,frmt)
    implicit none
    character(*) :: str
    character(*) :: frmt
    integer :: mtrx(n,n)
    integer :: n
    integer :: i
    integer :: j
    write(6,'(a)') str
    do i=1,n
      write(6,frmt) (mtrx(i,j),j=1,n)
    end do
    return
  end subroutine wtmtrxif

  subroutine wtmtrxrf(str,mtrx,n,frmt)
    implicit none
    character(*) :: str
    character(*) :: frmt
    real(4) :: mtrx(n,n)
    integer :: n
    integer :: i
    integer :: j
    write(6,'(a)') str
    do i=1,n
      write(6,frmt) (mtrx(i,j),j=1,n)
    end do
    return
  end subroutine wtmtrxrf

  subroutine wtmtrxdf(str,mtrx,n,frmt)
    implicit none
    character(*) :: str
    character(*) :: frmt
    real(8) :: mtrx(n,n)
    integer :: n
    integer :: i
    integer :: j
    write(6,'(a)') str
    do i=1,n
      write(6,frmt) (mtrx(i,j),j=1,n)
    end do
    return
  end subroutine wtmtrxdf

  subroutine matinv3r(a,b,m,det)
    implicit none
    real(4), dimension(3,3) :: a
    real(4), dimension(3) :: b
    integer :: m
    real(4) :: det
    real(4) :: pivot(3)
    integer :: ipivot(3)
    integer :: index(3,2)
    integer :: ierr
    !call GAUSSJ (a, 3,  b, ierr)
    !det=determ3r1(a)
    call matinvr(a,3,3,b,m,det,ipivot,index,pivot)
    return
  end subroutine matinv3r

  subroutine matinv3d(a,b,m,det)
    implicit none
    real(8), dimension(3,3) :: a
    real(8), dimension(3) :: b
    integer :: m
    real(8) :: det
    real(8) :: pivot(3)
    integer :: ipivot(3)
    integer :: index(3,2)
    integer :: ierr
    !call GAUSSJ (a, 3,  b, ierr)
    !det=determ3r1(a)
    call matinvd(a,3,3,b,m,det,ipivot,index,pivot)
    return
  end subroutine matinv3d


  ! real(4) version
  subroutine matinv6r(a,b,m,det)
    implicit none
    real(4), dimension(6,6) :: a
    real(4), dimension(6) :: b
    integer :: m
    real(4) :: det
    real(4) :: pivot(6)
    integer :: ipivot(6)
    integer :: index(6,2)
    integer :: ierr
    !call GAUSSJr(a, 6,  b, ierr)
    call matinvr(a,6,6,b,m,det,ipivot,index,pivot)
    return
  end subroutine matinv6r

  ! real(8) version
  subroutine matinv6d(a,b,m,det)
    implicit none
    real(8),dimension(6,6) :: a
    real(8),dimension(6) :: b
    integer :: m
    real(8) :: det
    real(8) :: pivot(6)
    integer :: ipivot(6)
    integer :: index(6,2)
    integer :: ierr
    call GAUSSJd(a, 6,  b, ierr)
    !call matinvd(a,6,6,b,m,det,ipivot,index,pivot)
    return
  end subroutine matinv6d

  subroutine matinvr(a,nm,n,b,m,determ,ipivot,index,pivot)
    implicit none
    real(4) :: amax
    real(4) :: determ
    integer:: i
    integer:: icolum
    integer:: irow
    integer:: j
    integer:: jcolum
    integer:: jrow
    integer:: k
    integer:: l
    integer:: l1
    integer:: m
    integer:: n
    integer:: nm
    real(4) :: swap
    real(4) :: t
    !
    ! n*n matrix a is replaced by its inverse matrix a**-1
    ! if m is nonzero vector b is replaced by a**-1*b
    !
    real(4), dimension(nm,nm) :: a
    real(4), dimension(nm) :: b
    real(4), dimension(nm) ::pivot
    integer :: ipivot(nm)
    integer :: index(nm,2)
    equivalence (irow,jrow),(icolum,jcolum),(amax,t,swap)
    determ=1.0
    do j=1,n
      ipivot(j)=0
    end do
    do i=1,n
      amax=0.0
      do j=1,n
        if(ipivot(j)==1) cycle !go to 105
        do k=1,n
          ! if(ipivot(k)-1) 80,100,740
          if(ipivot(k)-1<0) then
            !80 continue
            if(abs(amax)>=abs(a(j,k))) cycle !go to 100
            irow=j
            icolum=k
            amax=a(j,k)
          else if(ipivot(k)-1>0) then
            return  !go to 740
          end if
        !100     continue
        end do
      !105   continue
      end do
      ipivot(icolum)=ipivot(icolum)+1
      !if(irow==icolum) go to 260
      if(irow/=icolum) then !go to 260
        determ=-determ
        do l=1,n
          swap=a(irow,l)
          a(irow,l)=a(icolum,l)
          a(icolum,l)=swap
        end do
        !if(m==0) go to 260
        if(m/=0) then !go to 260
          swap=b(irow)
          b(irow)=b(icolum)
          b(icolum)=swap
        end if
      end if
      !260 continue
      index(i,1)=irow
      index(i,2)=icolum
      pivot(i)=a(icolum,icolum)
      determ=determ*pivot(i)
      a(icolum,icolum)=1.0
      do l=1,n
        a(icolum,l)=a(icolum,l)/pivot(i)
      end do
      !if(m==0) go to 380
      if(m/=0) then  !go to 380
        b(icolum)=b(icolum)/pivot(i)
      end if
      !380 continue
      do l1=1,n
        if(l1==icolum) cycle  !go to 550
        t=a(l1,icolum)
        a(l1,icolum)=0.0
        do l=1,n
          a(l1,l)=a(l1,l)-a(icolum,l)*t
        end do
        !if(m==0) go to 550
        if(m/=0) then !go to 550
          b(l1)=b(l1)-b(icolum)*t
        end if
      !550   continue
      end do
    end do

    do i=1,n
      l=n+1-i
      if(index(l,1)==index(l,2)) cycle  !go to 710
      jrow=index(l,1)
      jcolum=index(l,2)
      do k=1,n
        swap=a(k,jrow)
        a(k,jrow)=a(k,jcolum)
        a(k,jcolum)=swap
      end do
    !710 continue
    end do
    !740 continue
    return 
  end subroutine matinvr

  subroutine matinvd(a,nm,n,b,m,determ,ipivot,index,pivot)
    implicit none
    real(8):: amax
    real(8):: determ
    integer:: i
    integer:: icolum
    integer:: irow
    integer:: j
    integer:: jcolum
    integer:: jrow
    integer:: k
    integer:: l
    integer:: l1
    integer:: m
    integer:: n
    integer:: nm
    real(8):: swap
    real(8):: t
    !
    ! n*n matrix a is replaced by its inverse matrix a**-1
    ! if m is nonzero vector b is replaced by a**-1*b
    !
    real(8),dimension(nm,nm) :: a
    real(8),dimension(nm) :: b
    real(8),dimension(nm) ::pivot
    integer :: ipivot(nm)
    integer :: index(nm,2)
    equivalence (irow,jrow),(icolum,jcolum),(amax,t,swap)
    determ=1.0
    do j=1,n
      ipivot(j)=0
    end do
    do i=1,n
      amax=0.0
      do j=1,n
        if(ipivot(j)==1) cycle !go to 105
        do k=1,n
          ! if(ipivot(k)-1) 80,100,740
          if(ipivot(k)-1<0) then
            !80 continue
            if(abs(amax)>=abs(a(j,k))) cycle !go to 100
            irow=j
            icolum=k
            amax=a(j,k)
          else if(ipivot(k)-1>0) then
            !go to 740
            return
          end if
        !100     continue
        end do
      !105   continue
      end do
      ipivot(icolum)=ipivot(icolum)+1
      !if(irow==icolum) go to 260
      if(irow/=icolum) then
        determ=-determ
        do l=1,n
          swap=a(irow,l)
          a(irow,l)=a(icolum,l)
          a(icolum,l)=swap
        end do
        !if(m==0) go to 260
        if(m/=0) then !go to 260
          swap=b(irow)
          b(irow)=b(icolum)
          b(icolum)=swap
        end if
      end if
      !260 continue
      index(i,1)=irow
      index(i,2)=icolum
      pivot(i)=a(icolum,icolum)
      determ=determ*pivot(i)
      a(icolum,icolum)=1.0
      do l=1,n
        a(icolum,l)=a(icolum,l)/pivot(i)
      end do
      !if(m==0) go to 380
      if(m/=0) then !go to 380
        b(icolum)=b(icolum)/pivot(i)
      end if
      !380 continue
      do l1=1,n
        !if(l1==icolum) go to 550
        if(l1/=icolum) cycle  !go to 550
        t=a(l1,icolum)
        a(l1,icolum)=0.0
        do l=1,n
          a(l1,l)=a(l1,l)-a(icolum,l)*t
        end do
        !if(m==0) go to 550
        if(m/=0) then  !go to 550
          b(l1)=b(l1)-b(icolum)*t
        end if
      !550   continue
      end do
    end do
    do i=1,n
      l=n+1-i
      if(index(l,1)==index(l,2)) cycle !go to 710
      jrow=index(l,1)
      jcolum=index(l,2)
      do k=1,n
        swap=a(k,jrow)
        a(k,jrow)=a(k,jcolum)
        a(k,jcolum)=swap
      end do
    !710 continue
    end do
    !740 continue
    return
  end subroutine matinvd

  !  subroutine GAUSSJ (a, n, np, b, m, mp, ierr)
  !  Purpose: Solution of the system of linear equations AX = B by
  !     Gauss-Jordan elimination, where A is a matrix of order N and B is
  !     an N x M matrix.  On output A is replaced by its matrix inverse
  !     and B is preplaced by the corresponding set of solution vectors.

  !  Source: W.H. Press et al, "Numerical Recipes," 1989, p. 28.

  !  Modifications:
  !     1. Double  precision.
  !     2. Error parameter IERR included.  0 = no error. 1 = singular
  !        matrix encountered; no inverse is returned.

  !  Prepared by J. Applequist, 8/17/91.
  !IMPLICIT real (8)(a - h, o - z)
  !IMPLICIT real (a - h, o - z)
  !    Set largest anticipated value of N.
  !PARAMETER (nmax = 500)

  subroutine GAUSSJr(a, n,  b, ierr)
    implicit none
    real(4) :: big
    real(4) :: dum
    integer:: l
    integer:: ll
    real(4) :: pivinv
    real(4) :: a (n, n)
    real(4) :: b (n)
    integer :: n
    integer :: ierr
    integer :: i
    integer :: j
    integer :: k
    integer,allocatable :: ipiv (:)
    integer,allocatable :: indxr (:)
    integer,allocatable :: indxc (:)
    integer :: irow
    integer :: icol
    allocate(ipiv (n), indxr (n),indxc (n))
    ierr = 0
    DO j = 1, n
      ipiv (j) = 0
    !11  continue
    end DO
    DO i = 1, n
      big = 0
      DO j = 1, n
        IF (ipiv (j) /= 1) then
          DO k = 1, n
            IF (ipiv (k) == 0) then
              IF (abs (a (j, k) ) >= big) then
                big = abs (a (j, k) )
                irow = j
                icol = k
              endIF
            ELSEIF (ipiv (k) >1) then
              ierr = 1
              RETURN
            endIF
          !12        continue
          end DO
        endIF
      !13    continue
      end DO
      ipiv (icol) = ipiv (icol) + 1
      IF (irow/=icol) then
        DO l = 1, n
          dum = a (irow, l)
          a (irow, l) = a (icol, l)
          a (icol, l) = dum
        !14      continue
        end DO
        !DO l = 1, m
        dum = b (irow)
        b (irow) = b (icol)
        b (icol) = dum
      !15    continue
         !end DO
      endIF
      indxr (i) = irow
      indxc (i) = icol
      IF (a (icol, icol) == 0) then
        ierr = 1
        RETURN
      endIF
      pivinv = 1 / a (icol, icol)
      a (icol, icol) = 1
      DO l = 1, n
        a (icol, l) = a (icol, l) * pivinv
      !16  continue
      end DO
      !DO l = 1, m
      b (icol) = b (icol) * pivinv
      !17 continue
      !end DO
      DO ll = 1, n
        IF (ll/=icol) then
          dum = a (ll, icol)
          a (ll, icol) = 0
          DO l = 1, n
            a (ll, l) = a (ll, l) - a (icol, l) * dum
          !18     continue
          end DO
          !DO l = 1, m
          b (ll) = b (ll) - b (icol) * dum
        !19   continue
           !end DO
        endIF
      !21 continue
      end DO
    !22 continue
    end DO
    DO l = n, 1, - 1
      IF (indxr (l) /=indxc (l) ) then
        DO k = 1, n
          dum = a (k, indxr (l) )
          a (k, indxr (l) ) = a (k, indxc (l) )
          a (k, indxc (l) ) = dum
        !23     continue
        end DO
      endIF
    !24 continue
    end DO
    RETURN
  end subroutine GAUSSJr

  subroutine GAUSSJd(a, n,  b, ierr)
    implicit none
    real(8):: big
    real(8):: dum
    integer:: l
    integer:: ll
    real(8):: pivinv
    real(8) :: a (n, n)
    real(8) :: b (n)
    integer :: n
    integer :: ierr
    integer :: i
    integer :: j
    integer :: k
    integer,allocatable :: ipiv (:)
    integer,allocatable :: indxr (:)
    integer,allocatable :: indxc (:)
    integer :: irow
    integer :: icol
    allocate(ipiv (n), indxr (n),indxc (n))
    ierr = 0
    DO j = 1, n
      ipiv (j) = 0
    !11 continue
    end DO
    DO i = 1, n
      big = 0
      DO j = 1, n
        IF (ipiv (j) /= 1) then
          DO k = 1, n
            IF (ipiv (k) == 0) then
              IF (abs (a (j, k) ) >= big) then
                big = abs (a (j, k) )
                irow = j
                icol = k
              endIF
            ELSEIF (ipiv (k) >1) then
              ierr = 1
              RETURN
            endIF
          !12       continue
          end DO
        endIF
      !13   continue
      end DO
      ipiv (icol) = ipiv (icol) + 1
      IF (irow/=icol) then
        DO l = 1, n
          dum = a (irow, l)
          a (irow, l) = a (icol, l)
          a (icol, l) = dum
        !14     continue
        end DO
        !DO l = 1, m
        dum = b (irow)
        b (irow) = b (icol)
        b (icol) = dum
      !15   continue
         !end DO
      endIF
      indxr (i) = irow
      indxc (i) = icol
      IF (a (icol, icol) == 0) then
        ierr = 1
        RETURN
      endIF
      pivinv = 1 / a (icol, icol)
      a (icol, icol) = 1
      DO l = 1, n
        a (icol, l) = a (icol, l) * pivinv
      !16 continue
      end DO
      !DO l = 1, m
      b (icol) = b (icol) * pivinv
      !17 continue
      !end DO
      DO ll = 1, n
        IF (ll/=icol) then
          dum = a (ll, icol)
          a (ll, icol) = 0
          DO l = 1, n
            a (ll, l) = a (ll, l) - a (icol, l) * dum
          !18     continue
          end DO
          !DO l = 1, m
          b (ll) = b (ll) - b (icol) * dum
        !19   continue
           !end DO
        endIF
      !21 continue
      end DO
    !22 continue
    end DO
    DO l = n, 1, - 1
      IF (indxr (l) /=indxc (l) ) then
        DO k = 1, n
          dum = a (k, indxr (l) )
          a (k, indxr (l) ) = a (k, indxc (l) )
          a (k, indxc (l) ) = dum
        !23     continue
        end DO
      endIF
    !24 continue
    end DO
    RETURN
  end subroutine GAUSSJd

  subroutine wtrvct6(str,ej1)
    implicit none
    integer:: j
    character(*) :: str
    real(4) :: ej1(6)
    write(6,'(a,6f10.5)') str,(ej1(j),j=1,6)
    return
  end subroutine wtrvct6

  subroutine wtrmtrx6(str,rmtrx)
    implicit none
    integer:: i
    integer:: j
    character(*) :: str
    real(4) :: rmtrx(6,6)
    write(6,*) str
    do i=1,6
      write(6,'(6f10.5)') (rmtrx(i,j),j=1,6)
    end do
    return
  end subroutine wtrmtrx6

  subroutine wtimtrx6(str,rmtrx)
    implicit none
    integer:: i
    integer:: j
    character(*) :: str
    integer :: rmtrx(6,6)
    write(6,*) str
    do i=1,6
      write(6,'(6i5)') (rmtrx(i,j),j=1,6)
    end do
    return
  end subroutine wtimtrx6

  subroutine wtrmtrx3(str,rmtrx)
    implicit none
    integer:: i
    integer:: j
    character(*) :: str
    real(4) :: rmtrx(3,3)
    write(6,*) str
    do i=1,3
      write(6,'(3f10.5)') (rmtrx(i,j),j=1,3)
    end do
    return
  end subroutine wtrmtrx3

  subroutine wtimtrx3(str,rmtrx)
    implicit none
    integer:: i
    integer:: j
    character(*) :: str
    integer :: rmtrx(3,3)
    write(6,*) str
    do i=1,3
      write(6,'(3i5)') (rmtrx(i,j),j=1,3)
    end do
    return
  end subroutine wtimtrx3


  logical function iszerovr0(x,n)
    implicit none
    integer:: i
    integer:: n
    real(4) :: x(n)
    do i=1,n
      if(x(i)/=0) then
        iszerovr0=.false.
        return
      end if
    end do
    iszerovr0=.true.
  end function iszerovr0

  logical function iszerovr1(x,n,eps)
    implicit none
    integer:: i
    integer:: n
    real(4) :: eps
    real(4) :: x(n)
    do i=1,n
      if(abs(x(i))>eps) then
        iszerovr1=.false.
        return
      end if
    end do
    iszerovr1=.true.
  end function iszerovr1

  logical function iszerovi(x,n)
    implicit none
    integer:: i
    integer:: n
    integer :: x(n)
    do i=1,n
      if(x(i)/=0) then
        iszerovi=.false.
        return
      end if
    end do
    iszerovi=.true.
  end function iszerovi

  real(4) function atanfr(x)
    implicit none
    real(4) :: pi
    ! t: angle in rad/6.28318   (0<=t<=1)
    real(4) :: x(2)
    real(4) :: t
    !pi=acos(-1.0)
    pi=atan(1.0)*4
    t=atan2(x(2),x(1))/(2*pi)  ! -0.5<=t<=0.5
    if(t<0) then
      t=t+1                   ! 0<=t<=1
    end if
    atanfr=t
  end function atanfr

  real(4) function determ3r(u1,u2,u3)
    implicit none
    real(4) :: u1(3)
    real(4) :: u2(3)
    real(4) :: u3(3)
    determ3r=u1(1)*(u2(2)*u3(3)-u2(3)*u3(2))+&
      &    u1(2)*(u2(3)*u3(1)-u2(1)*u3(3))+&
      &    u1(3)*(u2(1)*u3(2)-u2(2)*u3(1))
    return
  end function determ3r

  real(4) function determ3r1(u)
    implicit none
    real(4) :: u(3,3)
    determ3r1=u(1,1)*(u(2,2)*u(3,3)-u(2,3)*u(3,2))+&
      &    u(1,2)*(u(2,3)*u(3,1)-u(2,1)*u(3,3))+&
      &    u(1,3)*(u(2,1)*u(3,2)-u(2,2)*u(3,1))
    return
  end function determ3r1

  integer function lcm(m,n) ! calculate least common multiple
    implicit none
    integer :: m;integer :: n;
    integer :: i
    do i=1,m*n
      if(mod(i,m)==0 .and. mod(i,n)==0) then
        lcm =i
        return
      end if
    end do
    lcm=1
  end function lcm

  integer function gcd(m0,n0) ! greatest common divisor of m and n
    implicit none
    integer:: m0
    integer:: n0
    integer:: k
    integer :: m
    integer :: n
    integer :: sm
    integer :: sn
    if(m0 ==0) then
      gcd=n0
      return
    end if

    m=abs(m0)
    n=abs(n0)
    IF(m > 0 .AND. n > 0) THEN
      DO
        k = MOD(m, n)
        IF(k==0) EXIT
        m = n
        n = k
      end DO
      IF(n>1)THEN
        gcd = n
      ELSE
        gcd = 1
      end IF
    else
      gcd=1
    end IF
  end function gcd
end module fmath
