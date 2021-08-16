function [P, spl, options]  = spherical_wave_sim(c, U0, phi, N, mono_loc, pos_vec, freq, samp_freq, t_end)
% This function analytically simulates the 3D temporal pressure field for arbitrary number of monopoles.
%
%[Pressure Field, options] = monopole_sim(rho, c, U0, a, n, mono_loc, field, field_points, freq, samp_freq, t_end)
% ------------------------------------------------------------------------------------------------------------------
% Inputs
%
% c         --> Speed of sound in medium [m/s] - [1 x 1]
% U0        --> Amplitude vector for N monopoles  [m/s] - [1 x N]
% Phi       --> Phase shift for N monopoles [rad] - [1 x N]
% N         --> Number of monopoles - [1 x 1]
% mono_loc  --> Location of Monopoles in 3D [m] - [x_mono; y_mono; z_mono]- [3 x N] 
% pos_vec   --> Position vector for pressure computation [m] - [x_vec; y_vec; z_vec]' - [3 x p]
% freq      --> Excitation frequencies for N monopoles [Hz] - [1 x n]
% samp_freq --> Sampling frequency of the temporal signal [Hz] - [1 x 1]
% t_end     --> Final time [s] (Set t_end = 0, if only Amplitude is required)
% --------------------------------------------------------------------------------------------------------------------
% Outputs
%
% P         --> Pressure Field
% spl       --> Sound Pressure Level Field
% options   --> Extra stuff for analysis
% --------------------------------------------------------------------------------------------------------------------
% Reference - 'Fundamentals of Acoustics', Kinsler, Frey, Coppens and Sanders - 3rd Edition, Wiley Publications
% Version - 1
% Copyright - Revant Adlakha (2021)
% Sound and Vibrations Lab, UB

time = 0:1/samp_freq:t_end;
ang_freq = 2*pi*freq;
size_pos_vec = size(pos_vec);
size_pos_vec = size_pos_vec(1);

P = zeros(size_pos_vec, length(time));
p = zeros(size_pos_vec, length(time));

for t = 1:length(time) % Change this to 'parfor' if there is enough memory avaiable and a large grid size is choosen or switch to 'for'
    for n = 1:N
        r = sqrt((pos_vec(:, 1) - mono_loc(1, n)).^2 + (pos_vec(:, 2) - mono_loc(2, n)).^2 + (pos_vec(:, 3) - mono_loc(3, n)).^2);
        kappa = ang_freq(n)/c;
        
        p(:, t) = p(:, t) + U0(n)*exp(1i*(ang_freq(n)*time(t) - kappa*r + phi(n)))./r;
        %p(:, t) = p(:, t) + U0*exp(1i*(ang_freq(n)*time(t) - kappa*r + phi(n)))./r;
    end
    P(:, t) = P(:, t) + p(:, t);
end
P_ref = 20e-6; % Reference Pressure
spl = real(20*log10(P/P_ref));

% Passing extra variables back to the main code
options.p = p;
options.ang_freq = ang_freq;
options.spl = real(20*log10(P/P_ref)); % Sound Pressure Level
options.time = time;
end


