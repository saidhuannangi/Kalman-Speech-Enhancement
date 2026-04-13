clc;
clear;
close all;

%% 🎙️ 1. Load Clean Speech
load handel
x = y;
fs = Fs;  % Place file in same folder
x = x(:,1); % Convert to mono

t = (0:length(x)-1)/fs;

%% 🔊 2. Add Noise
% Add Noise
noise = 0.3 * randn(size(x));   % Strong noise
noisy = x + noise;      % White Gaussian Noise
noisy = x + noise;

%% 📊 3. Initialize Kalman Filter Parameters
N = length(noisy);

x_est = zeros(N,1);   % Estimated signal
P = 1;                % Error covariance

A = 1;                % State transition
H = 1;                % Observation model
Q = 1e-5;             % Process noise covariance
R = 0.01;             % Measurement noise covariance

%% 🔁 4. Kalman Filtering
for k = 2:N
    
    % Prediction
    x_pred = A * x_est(k-1);
    P_pred = A * P * A' + Q;
    
    % Kalman Gain
    K = P_pred * H' / (H * P_pred * H' + R);
    
    % Update
    x_est(k) = x_pred + K * (noisy(k) - H * x_pred);
    P = (1 - K * H) * P_pred;
end

%% 📈 5. Performance Metrics

% SNR Calculation
snr_noisy = 10 * log10(sum(x.^2) / sum((x - noisy).^2));
snr_filtered = 10 * log10(sum(x.^2) / sum((x - x_est).^2));

% MSE Calculation
mse_noisy = mean((x - noisy).^2);
mse_filtered = mean((x - x_est).^2);

fprintf('\n--- PERFORMANCE RESULTS ---\n');
fprintf('SNR (Noisy)     : %.2f dB\n', snr_noisy);
fprintf('SNR (Filtered)  : %.2f dB\n', snr_filtered);
fprintf('MSE (Noisy)     : %.6f\n', mse_noisy);
fprintf('MSE (Filtered)  : %.6f\n', mse_filtered);

%% 📊 6. Plot Waveforms

figure;

subplot(3,1,1);
plot(t, x);
title('Original Clean Speech');
xlabel('Time (s)');
ylabel('Amplitude');

subplot(3,1,2);
plot(t, noisy);
title('Noisy Speech');
xlabel('Time (s)');
ylabel('Amplitude');

subplot(3,1,3);
plot(t, x_est);
title('Enhanced Speech using Kalman Filter');
xlabel('Time (s)');
ylabel('Amplitude');

%% 🔥 7. Spectrograms (HIGH MARKS SECTION)

figure;

subplot(3,1,1);
spectrogram(x, 256, 200, 256, fs, 'yaxis');
title('Original Speech Spectrogram');

subplot(3,1,2);
spectrogram(noisy, 256, 200, 256, fs, 'yaxis');
title('Noisy Speech Spectrogram');

subplot(3,1,3);
spectrogram(x_est, 256, 200, 256, fs, 'yaxis');
title('Enhanced Speech Spectrogram');

%% 🎧 8. Play Audio (Optional)
% sound(x, fs); pause(3);
% sound(noisy, fs); pause(3);
% sound(x_est, fs);

%% 💾 9. Save Output
audiowrite('enhanced_kalman.wav', x_est, fs);
disp('Original Sound');
sound(x, fs);
pause(length(x)/fs);

disp('Noisy Sound');
sound(noisy, fs);
pause(length(x)/fs);

disp('Filtered Sound');
sound(x_est, fs);