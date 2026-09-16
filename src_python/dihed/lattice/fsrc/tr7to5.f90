subroutine tr7to5e(ei,fi)
    !       for dodecagonal
    real fi(*),ei(*)
    integer sd(7,7)
    data sd / 0, 1, 0, 0, 0,-1, 0,&
      &          1, 0, 1, 0, 0, 0, 0,&
      &          0, 1, 0, 1, 0, 0, 0,&
      &          0, 0, 1, 0, 1, 0, 0,&
      &          1, 0,-1, 0, 1, 0, 0,&
      &          0, 1, 0,-1, 0, 1, 0,&
      &          0, 0, 0, 0, 0, 0, 1/

    do j=1,6
      fi(j)=0.
      if(j<=4) then
        do k=1,6
          fi(j)=fi(j)+sd(k,j)*ei(k)
           !     fi(j)=fi(j)+sd(k,j)*ei(k)/3
        end do
      else if (j==5) then
        fi(5)=ei(7)
        fi(6)=0
      end if
    end do
    return
  END subroutine tr7to5e

  subroutine tr7to5i(ei,fi)
    !       for dodecagonal
    integer fi(*),ei(*)
    integer sd(7,7)
    data sd / 1, 0, 0, 0,-1, 0, 0,&
      &          0, 1, 0, 0, 0,-1, 0,&
      &          0, 0, 1, 0, 1, 0, 0,&
      &          0, 0, 0, 1, 0, 1, 0,&
      &          1, 0,-1, 0, 1, 0, 0,&
      &          0, 1, 0,-1, 0, 1, 0,&
      &          0, 0, 0, 0, 0, 0, 1/
  
    do j=1,6
      fi(j)=0.
      if(j<=4) then
        do k=1,6
          fi(j)=fi(j)+sd(k,j)*ei(k)
        end do
      else if (j==5) then
        fi(5)=ei(7)
        fi(6)=0
      end if
    end do
    return
  END subroutine tr7to5i
