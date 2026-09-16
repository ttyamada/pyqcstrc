  subroutine tr6to5e(ei,fi)
    real fi(*),ei(*)
    integer sd(6,6)
    data sd/ &
      &       4,-1,-1,-1, 0, 0,&
      &      -1, 4,-1,-1, 0, 0,&
      &      -1,-1, 4,-1, 0, 0,&
      &      -1,-1,-1, 4, 0, 0,&
      &      -1,-1,-1,-1, 0, 0,&
      &       0, 0, 0, 0, 5, 0/
    do j=1,6
      fi(j)=0.
      do k=1,6
        fi(j)=fi(j)+sd(j,k)*ei(k)/5
      end do
    end do
    return
  END subroutine tr6to5e

  subroutine tr6to5i(ei,fi)
    integer fi(*),ei(*)
    integer sd(6,6)
    data sd/ &
      &         1, 0, 0, 0, 0, 0,&
      &         0, 1, 0, 0, 0, 0,&
      &         0, 0, 1, 0, 0, 0,&
      &         0, 0, 0, 1, 0, 0,&
      &        -1,-1,-1,-1, 0, 0,&
      &         0, 0, 0, 0, 1, 0/
    do j=1,6
      fi(j)=0.
      do k=1,6
        fi(j)=fi(j)+sd(j,k)*ei(k)
      end do
    end do
    return
  END subroutine tr6to5i
  