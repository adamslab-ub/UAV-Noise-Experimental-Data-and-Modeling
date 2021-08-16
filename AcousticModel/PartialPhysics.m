% Initializes the parameters for the partial physics model and calls it

function [ spl_mic] = PartialPhysics(par_U, pos_vec, param)

[r,~] = size(pos_vec);
spl_mic = zeros(r,1);


for i = 1:r
    param.pos_vec = pos_vec(i,:);
    U0 = par_U(i,:);
    % Calling the monopole_sim function to generate the pressure field
    [P_i, spl_mic_i, options] = spherical_wave_sim(param.c, U0, param.phi, param.n, param.mono_loc, param.pos_vec, param.freq, param.samp_freq, param.t_end);

    spl_mic(i,1) = spl_mic_i;
end  


end