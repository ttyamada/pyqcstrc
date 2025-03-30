module qsort
  implicit none

contains

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

end module