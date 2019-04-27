using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Test
{
    public interface Observer
    {
        int StartDate { get; set; }
        void Message(Temperature temperature);
    }

    public class Nurse : Observer
    {
        public int StartDate { get => throw new NotImplementedException(); set => throw new NotImplementedException(); }

        public void Message(Temperature temperature)
        {
            Console.WriteLine(temperature.ToString());
        }
    }
}
