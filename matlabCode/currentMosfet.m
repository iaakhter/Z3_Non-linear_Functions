function [I, firDerIn, firDerOut, secDerIn, secDerOut] = currentMosfet(in, out, Vtp, Vtn, Vdd, Kn, Sn)
  if(nargin < 7) Sn = 1; end;
  if(nargin < 6) Kn = 1; end;
  if(nargin < 5) Vdd = 1; end;
  if(nargin < 4) Vtn = 0.25; end;
  if(nargin < 3) Vtp = -0.25; end;
  Kp = -Kn/2.0;
  Sp = Sn*2.0;
  In = 0.0;

  firDerInn = 0.0;
  firDerOutn = 0.0;
  secDerInn = 0.0;
  secDerOutn = 0.0;
  if (in <= Vtn)
  	In = 0.0;
    firDerInn = 0.0;
    firDerOutn = 0.0;
    secDerInn = 0.0;
    secDerOutn = 0.0;
  elseif (Vtn <= in && in <= out + Vtn)
    In = Sn*(Kn/2.0)*(in - Vtn)*(in - Vtn);
    firDerInn = Sn*Kn*(in - Vtn);
    firDerOutn = 0.0;
    secDerInn = Sn*Kn;
    secDerOutn = 0.0;
  elseif (in >= out + Vtn)
    In = Sn*(Kn)*(in - Vtn - out/2.0)*out;
    firDerInn = Sn*Kn*out;
    firDerOutn = -Sn*Kn*out;
    secDerInn = 0.0;
    secDerOutn = -Sn*Kn;
  end;
  
  Ip = 0.0;
  firDerInp = 0.0;
  firDerOutp = 0.0;
  secDerInp = 0.0;
  secDerOutp = 0.0;
  if (in - Vtp >= Vdd)
  	Ip = 0.0;
    firDerInp = 0.0;
    firDerOutp = 0.0;
    secDerInp = 0.0;
    secDerOutp = 0.0;
  elseif (out <= in - Vtp && in - Vtp <= Vdd )
    Ip = Sp*(Kp/2.0)*(in - Vtp - Vdd)*(in - Vtp - Vdd);
    firDerInp = Sp*Kp*(in - Vtp - Vdd);
    firDerOutp = 0.0;
    secDerInp = Sp*Kp;
    secDerOutp = 0.0;
  elseif (in - Vtp <= out)
    Ip = Sp*Kp*((in - Vtp - Vdd) - (out - Vdd)/2.0)*(out - Vdd);
    firDerInp = Sp*Kp*(out - Vdd);
    firDerOutp = -Sp*Kp*(out - Vdd);
    secDerInp = 0.0;
    secDerOutp = -Sp*Kp;
  end;

  I = -(In + Ip);
  firDerIn = -(firDerInn + firDerInp);
  firDerOut = -(firDerOutn + firDerOutp);
  secDerIn = -(secDerInn + secDerInp);
  secDerOut = -(secDerOutn + secDerOutp);
end % inverter
