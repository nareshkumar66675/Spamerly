using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Test
{
    public enum Unit
    {
        K, C, F
    }

    public enum BodyLocation
    {
        ArmPit,
        UnderTongue
    }
    public class Temperature : TemperatureNotifier
    {

        public Unit TemperatureUnit { get;  }

        private float _temp;

        public float Temp
        {
            get
            {
                return _temp;
            }
            set
            {
                this._temp = value;

            }
        }

        protected float ConvertTemperature(Unit unit)
        {
            return Temp;
        }


    }

    public enum Fever
    {
        HighFever,
        LowFever,
        Hypothermia,
        NoFever
    }

    public class BodyTemperature:Temperature
    {
        public BodyLocation Location { get; set; }

        public Fever CheckFever()
        {
            var temp = ConvertTemperature(Unit.F);

            if (temp > 101)
            {
                Notify(this);
                return Fever.HighFever;
            }
            else if (temp > 98 && temp < 101)
            {
                return Fever.LowFever;
            }
            else if (temp > 96 && temp < 98)
                return Fever.NoFever;
            else
                return Fever.Hypothermia;
        }
    }

    public class TemperatureNotifier
    {
        public List<Observer> Observers { get; set; }

        public void Attach(Observer observer)
        {
            this.Observers.Add(observer);
        }

        public void Dettach(Observer observer)
        {
            this.Observers.Remove(observer);
        }

        public void Notify(Temperature temperature)
        {
            Observers.ForEach(observer => observer.Message(temperature));
        }
    }
}
