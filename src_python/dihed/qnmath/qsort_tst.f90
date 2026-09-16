
program qsort_tst
	use qsort

	real f(10)
	integer ip(10)
	integer s(10,0:1)

	do i=1,10
		f(i)=11-i
		write(6,'(*(g0,1x))') 'f=',f(i)
	end do

	write(6,*)
	call qsortr(f,s,ip,10)

	do i=1,10
		write(6,'(*(g0,1x))') 'f=',f(i)
	end do
end program