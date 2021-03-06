% The script:
% 1. Converts the poison generated by Python into MATLAB format
% 2. Runs the simulation
% 3. Saves the results
manual = false;
if manual
    att_idx = 1; % SET via a parameter
    n_poisons = 0; % IMPORTANT - set to the correct number when running manually
end
attack_gen_mode = 0; % IMPORTANT: Remember to set to 0 when running Poisoning.py 
period = 7.05;
if att_idx == 1
    att_len = 3.0;%1.35;%1.81;%5.5;
    att_level = 2.35;
    att_delay = 0;%0;
    poison_len = period * 3; 
elseif att_idx == 2 || att_idx == 3
    att_level = 7;
    att_delay = 0;
    poison_len = period * 2; % IMPORTANT: if the attack is delayed - we must add time to restore from it.
elseif att_idx == 31
    att_level = 22.9;%23.0;
    att_delay = 19.5; % for attack 31 generation with 4 poisons
    poison_len = period * 5; % IMPORTANT: if the attack is delayed - we must add time to restore from it.
end


for d = 2%:4
    att_poison_file  = sprintf("../te_attack_%d_poison.mat", att_idx);
    load(att_poison_file)

    fprintf('Poison loaded. num_poisons %d, att_idx %d\n', n_poisons, att_idx)
    if att_idx == 1
        s = size(p_xmeas_0);
    else
        s = size(p_xmeas_13);
    end
    
    pfactor = 3;
    for d0=1%att_len = 3%2.8:0.5:4.0
        for d1=1%att_level = 1.6:0.2:2.4
            for d2=1%att_delay = 0:0.25:8.0
                % fprintf('Poison factor %d\n', pfactor)
                % prepare the poison parts...
                poison_len = period * pfactor;

                % make sure the upper bound is set to the correct number of hours
                % number_of_poisons * poison_time_length (which is 14.10)
                times = linspace(0,poison_len * n_poisons, s(1,2));
                
                % we save the generated poison at a very high resolution
                plen_ticks = poison_len * 10000;
                if attack_gen_mode == 1 || attack_gen_mode == 2 
                    poisons = zeros( n_poisons, plen_ticks ); % does not matter
                else
                    np = s(2)/plen_ticks; % calculate from the loaded file size
                    poisons = zeros( np, plen_ticks );
                    for pi = 1:1:np
                        pstart = (pi-1)*plen_ticks + 1;
                        pend = pstart + plen_ticks - 1;
                        if att_idx == 1
                            poisons(pi,:) = p_xmeas_0( pstart : pend);
                        else
                            poisons(pi,:) = p_xmeas_13( pstart : pend);
                        end
                    end
                end

                fprintf('\nAtt_level %f att_delay %f\n', att_level, att_delay)
                fprintf('Running simulator!\n');

                % run the simulation
                if attack_gen_mode == 0
                    duration = string(poison_len*(n_poisons+3));
                elseif attack_gen_mode == 1
                    % I used to create the attack during the first poison period
                    % However, I uncountered the problem with holding attack (31) - the
                    % attack that worked during the first poison did not work after the
                    % required number of poisons. I've checked that the spares I took at
                    % the both ends of the poison helps stabilizing the system. Yet the
                    % poison did not work if not in the very beginning. Therefore, I
                    % switched to generate the poison after the desired number of poisons
                    % (can be guessed initially). When saving the attack, I still leave
                    % only the single poison_len time, not the entire timeline.
                    % duration = string(poison_len); %"14.1";
    %                  duration = string(poison_len*(n_poisons+1)+period*2); %"14.1";
                     duration = string(poison_len*(n_poisons+1)); %"14.1";
        %            duration = string(poison_len*2.5);
                elseif attack_gen_mode == 2
                    duration = string(poison_len*25);
                end
                % IF YOU WANT DRAWING or run manually, run this one 
                if manual
                    sim('MultiLoop_mode1.mdl');
                else
                    % IF YOU DON'T WANT DRAWING, run this one 
                    % I need to reset the stop function, because the drawing causes errors
                    sim_out = sim('MultiLoop_mode1.mdl', 'StopFcn', '', 'StopTime', duration);  

                    %retrieve the results
                    simout = sim_out.get('simout');
                    xmv = sim_out.get('xmv');
                end

                if attack_gen_mode == 0
%                     res_xmeas_file = sprintf("../TESimData_att1/TE_attack_%d_%d_%f_%f_%f_with_poison_var_xmeas.mat", att_idx, n_poisons, att_level, att_len, att_delay);
%                     res_xmvs_file = sprintf("../TESimData_att1/TE_attack_%d_%d_%f_%f_%f_with_poison_var_xmvs.mat", att_idx, n_poisons, att_level, att_len, att_delay);
                    res_xmeas_file = sprintf("../TE_attack_%d_with_poison_xmeas.mat", att_idx);
                    res_xmvs_file = sprintf("../TE_attack_%d_with_poison_xmvs.mat", att_idx);
                    fprintf('Simulator completed! Saving the results. (%s, %s)\n', res_xmeas_file, res_xmvs_file);
                    simout_len = size(simout);
                    fprintf("Last value of sep level %f time %f\n", simout(end,12), simout_len/100);
                    save(res_xmeas_file, 'simout');
                    save(res_xmvs_file, 'xmv');
                elseif attack_gen_mode == 1 
%                   res_xmeas_file = sprintf("../TE_attack_%d_%d_%f_%f_xmeas.mat", att_idx, n_poisons, att_level, att_delay );
%                   res_xmvs_file = sprintf("../TE_attack_%d_%d_%f_%f_xmvs.mat", att_idx, n_poisons, att_level, att_delay);   
                   res_xmeas_file = sprintf("../TE_attack_%d_xmeas.mat", att_idx);
                   res_xmvs_file = sprintf("../TE_attack_%d_xmvs.mat", att_idx);   
                   fprintf('Simulator completed! Saving the results. (%s, %s)\n', res_xmeas_file, res_xmvs_file);
                   simout_len = size(simout);
                   fprintf("Last value of sep level %f time %f\n", simout(end,12), simout_len/100);
                   if simout_len(1) > poison_len * 100
        %                 % Save only the last part with the attack
                        poison_len_ts_end = simout_len(1);
                        poison_len_ts_st = poison_len_ts_end - poison_len * 100 + 1;%round(poison_len * 100) * n_poisons + 1;
                        % Save only the first part with the attack
        %                 poison_len_ts_st = 1;
        %                 poison_len_ts_end = round(poison_len * 100);
                        %save simout - if you want to see it later
                        simout_copy = simout;
                        xmv_copy = xmv;
                        simout = simout(poison_len_ts_st : poison_len_ts_end, :);
                        xmv = xmv(poison_len_ts_st : poison_len_ts_end, :);
                        save(res_xmeas_file, 'simout');
                        save(res_xmvs_file, 'xmv');
                        simout = simout_copy;
                        xmv = xmv_copy;
                    else
                        save(res_xmeas_file, 'simout');
                        save(res_xmvs_file, 'xmv');
                   end
                elseif attack_gen_mode == 2
                    res_xmeas_file = sprintf("../TE_train_for_attack%d_xmeas.mat", att_idx);
                    res_xmvs_file = sprintf("../TE_train_for_attack%d_xmvs.mat", att_idx);
                    fprintf('Simulator completed! Saving the results. (%s, %s)\n', res_xmeas_file, res_xmvs_file);
                    simout_len = size(simout);
                    fprintf("Last value of sep level %f time %f\n", simout(end,12), simout_len/100);
    %                 save(res_xmeas_file, 'simout');
    %                 save(res_xmvs_file, 'xmv');
                end
            end
       end
    end
end
fprintf('Done.\n')


