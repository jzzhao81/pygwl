!>>> calculate hamk from hamr by fourier transformation
subroutine trans_hamr(nwan, nrpt, kvec, rvec, hamr, hamk)
  
  implicit none
  
  integer, parameter :: dp = 8
  real(8), parameter :: pi = acos(-1.0D0)
  
  ! external arguments
  ! number of Wannier orbitals
  integer, intent(in) :: nwan
  
  ! number of r-points
  integer, intent(in) :: nrpt
  
  ! r-point list
  real(kind=dp),intent(in) :: rvec(nrpt,3) 
  
  ! k-point list
  real(kind=dp), intent(in) :: kvec(3)
  
  ! hamiltonian in Wannier representation at r-space
  complex(dp), intent(in)  :: hamr(nrpt,nwan,nwan)
  
  ! hamiltonian in Wannier representation at k-space
  complex(dp), intent(out) :: hamk(nwan,nwan)
  
  ! local variables
  ! loop index for r-point
  integer :: irpt

  ! dummy variables
  real(dp) :: rdotk
  complex(dp) :: ratio

  ! initialize hamk
  hamk = dcmplx(0.0D0, 0.0D0)
  
  ! fourier transform hamr to hamk
  do irpt=1,nrpt
     rdotk = 2.0D0 * pi * dot_product( kvec(:), rvec(irpt,:))
     ratio = cmplx( cos(rdotk), sin(rdotk) )
     hamk = hamk + ratio * hamr(irpt, :,:)
  enddo ! over ikpt={1,nkpt} loop
  
end subroutine trans_hamr

!>>> calculate hamk from hamr by fourier transformation
!>>> for surface state
subroutine trans_hams(nwan, nrpt, zlay, rvec, kvec, hamr, hamk)
  
  implicit none
  
  integer, parameter :: dp = 8
  real(8), parameter :: pi = acos(-1.0D0)
  
  ! external arguments
  ! number of Wannier orbitals
  integer, intent(in) :: nwan
  
  ! number of r-points
  integer, intent(in) :: nrpt
  
  ! number of Z
  integer, intent(in) :: zlay
  
  ! r-point list
  real(dp), intent(in) :: rvec(nrpt,3)
  
  ! k-point list
  real(dp), intent(in) :: kvec(3)
  
  ! hamiltonian in Wannier representation at r-space
  complex(dp), intent(in)  :: hamr(nrpt,nwan,nwan)
  
  ! hamiltonian in Wannier representation at k-space
  complex(dp), intent(out) :: hamk(nwan*zlay, nwan*zlay)
  
  ! local variables
  ! loop index for r-point
  integer :: irpt

  ! iz
  integer :: ilay, jlay
  integer :: iwan, jwan, ii, jj

  ! dummy variables
  real(dp) :: rdotk
  complex(dp) :: ratio
  
  ! initialize hamk
  hamk = dcmplx(0.0D0, 0.0D0)
  
  ! fourier transform hamr to hamk
  do irpt=1,nrpt

     rdotk = 2.0D0 * pi * dot_product( kvec(1:2), rvec(irpt,1:2) )
     ratio = dcmplx( cos(rdotk), sin(rdotk) )

     do ilay = 1, zlay
        jlay = ilay + int(rvec(irpt,3))
        if( (jlay < 1) .or. (jlay > zlay)) cycle
        do iwan = 1, nwan
           ii = iwan + (ilay-1) * nwan
           do jwan = 1, nwan
              jj = jwan + (jlay-1) * nwan
              hamk(ii,jj) = hamk(ii,jj) + ratio * hamr(irpt,iwan,jwan)
           end do ! loop over jwan
        end do  ! loop over iwan
     end do ! loop over ilay

  enddo ! over irpt={1,nrpt} loop
 
end subroutine trans_hams

