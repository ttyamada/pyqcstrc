  ! real(4) version
  subroutine matinvnd(a,b,m,det)
    implicit none
    real(4),dimension(:,:) :: a
    real(4),dimension(:) :: b
    integer :: m
    real(4) :: det

    real(4),allocatable :: pivot(:)
    integer,allocatable :: ipivot(:)
    integer,allocatable :: index(:,:)  !index(6,2)
    integer :: ierr
    integer shp(2)
    shp=shape(a)
    n1=shp(1)
    allocate(pivot(n1),ipivot(n1),index(n1,2))
    call matinv0(a,n1,n1,b,m,det,ipivot,index,pivot)
    return
  end subroutine matinv6d

  subroutine matinv0(a,nm,n,b,m,determ,ipivot,index,pivot)
    implicit none
    real(4) :: a(nm,nm)
    real(4) :: b(nm)
    real(4) ::pivot(nm)

    integer :: ipivot(nm),index(nm,2)
    real(4) :: amax,determ
    integer:: i,icolum,irow,j,jcolum,jrow,k,l,l1
    integer:: m,n,nm
    real(4) :: swap
    real(4) :: t
    !
    ! n*n matrix a is replaced by its inverse matrix a**-1
    ! if m is nonzero vector b is replaced by a**-1*b
    !

    !equivalence (irow,jrow),(icolum,jcolum),(amax,t,swap)
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
        amax=a(l1,icolum) ! t=
        a(l1,icolum)=0.0
        do l=1,n
          a(l1,l)=a(l1,l)-a(icolum,l)*anax  !t
        end do
        !if(m==0) go to 550
        if(m/=0) then !go to 550
          b(l1)=b(l1)-b(icolum)*amax  !t
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
