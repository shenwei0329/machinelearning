% GNU Octave file (may also work with MATLAB(R) )
Fs=44100;minF=10;maxF=Fs/2;
sweepF=logspace(log10(minF),log10(maxF),200);
[h,w]=freqz([7.786928345734967e-03 1.557385669146993e-02 7.786928345734967e-03],[1 -1.735321984865454e+00 7.664696982483936e-01],sweepF,Fs);
semilogx(w,20*log10(h))
title('SoX effect: lowpass gain=0 frequency=1320 Q=0.707107 (rate=44100)')
xlabel('Frequency (Hz)')
ylabel('Amplitude Response (dB)')
axis([minF maxF -35 25])
grid on
disp('Hit return to continue')
pause
