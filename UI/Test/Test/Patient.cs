using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Test
{
    public class Person
    {
        protected Person()
        {
        }
        public string Name { get; set; }

        public int Age { get; set; }

        public string Address { get; set; }
    }

    public class Patient:Person
    {
        public int PatientId { get; set; }

        public Temperature BodyTemperature { get; set; }

        public List<Nurse> HandlingNurses { get; set; }


    }
}
