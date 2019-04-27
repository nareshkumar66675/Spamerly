using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Test
{
    class Program
    {
        public static int SetFour(int[] list)
        {
            List<int> pairs = new List<int>();

            Array.Sort(list);
            Dictionary<int, List<KeyValuePair<int, int>>> dict = new Dictionary<int, List<KeyValuePair<int, int>>>();

            int count = 0;

            for (int i = 0; i < list.Length; i++)
            {
                for (int j = i+1; j < list.Length; j++)
                {
                    int sum = list[i] + list[j];

                    if(dict.ContainsKey(0-sum))
                    {
                        dict.TryGetValue(0 - sum, out List<KeyValuePair<int, int>> value);

                        for(int l=0;l<value.Count;l++)
                        {
                            var pr = value[l];

                            if(pr.Key!=i && pr.Key!=j && pr.Value!=i&& pr.Value!=j)
                            {
                                if(list[pr.Key]+list[pr.Value]+list[i]+list[j]==0)
                                {
                                    count++;
                                }
                                    Console.WriteLine(list[pr.Key] +" "+list[pr.Value] + " " + list[i] + " " + list[j]);
                                break;
                            }
                        }
                    }

                    if(dict.ContainsKey(sum))
                    {
                        var value = dict.TryGetValue(sum,out List<KeyValuePair<int, int>> val);
                        if (val.Count > 0)
                            val.Add(new KeyValuePair<int, int>(i,j));
                        else
                        {
                            val = new List<KeyValuePair<int, int>>();
                            val.Add(new KeyValuePair<int, int>(i, j));
                        }
                    }
                    else
                    {
                        var val = new List<KeyValuePair<int, int>>();
                        val.Add(new KeyValuePair<int, int>(i, j));

                        dict.Add(sum, val);
                    }
                }
            }

            return count;
        }

        public static int MinCoins(int[] coins,int value)
        {
            int[] arr = new int[value + 1];

            

            arr[0] = 0;

            for(int i=1;i<arr.Length;i++)
            {
                arr[i] = int.MaxValue;
            }

            for (int i = 1; i <= value; i++)
            {
                for (int j = 0; j < coins.Length; j++)
                {
                    if (coins[j] <= i)
                    {
                        int res = arr[i - coins[j]];

                        if (res != int.MaxValue && res + 1 < arr[i])
                            arr[i]  = res +1;
                }
                }

            }
            

            return arr[value];
        }

        public static void Samp()
        {
            using (StreamReader reader = new StreamReader(Console.OpenStandardInput()))
                while (!reader.EndOfStream)
                {
                    string line = reader.ReadLine();

                    int value = int.Parse(line);

                    // Coins Face Value
                    int[] coins = { 1, 3, 5 };

                    int[] arrRecord = new int[value + 1];

                    arrRecord[0] = 0;

                    // Set the array to max value for comparison
                    for (int i = 1; i < arrRecord.Length; i++)
                    {
                        arrRecord[i] = int.MaxValue;
                    }

                    // To arrive at min no of coins for the given value
                    // For each increase in value, iterate and update the record with new coins count
                    for (int i = 1; i <= value; i++)
                    {
                        for (int j = 0; j < coins.Length; j++)
                        {
                            //If Coin value is less than or equal to Actual Value
                            if (coins[j] <= i)
                            { 
                                int tempValue = arrRecord[i - coins[j]];

                                //Update if the new value is less than the current value
                                if (tempValue != int.MaxValue && tempValue + 1 < arrRecord[i])
                                    arrRecord[i] = tempValue + 1;
                            }
                        }
                    }
                    Console.WriteLine(arrRecord[value]);
                }
        }

        public static void Check()
        {
            //using (StreamReader reader = new StreamReader(Console.OpenStandardInput()))
            //while (!reader.EndOfStream)
            //{
            string line = "Hello World"; //reader.ReadLine();
                    StringBuilder sbr = new StringBuilder();

                    string rslt = string.Empty;
            if (line == string.Empty) { }
                        //continue;
                    //else
                    //{
                    //    var words = line.Split(' ');
                        
                    //    for(int i = words.Length -1; i >= 0; i--)
                    //    {
                    //        sbr.Append(words[i]+" ");
                    //    }
                    //}

                    for(int i= line.Length-1; i>=0;i--)
                    {
                        if(line[i]==' ')
                {
                    sbr.Append(rslt+" ");
                    rslt = "";
                }
                        rslt = line[i]+rslt;
                    }
            sbr.Append(rslt);

                    Console.WriteLine(sbr);
                //}
        }

        public static int PalindromicNumber(int no,int itr)
        {
            int temp = no;

            int rev = 0;
            while(temp>0)
            {
                int rem = temp % 10;
                rev = rev * 10 + rem;
                temp = temp / 10;
            }

            if (no == rev)
                return itr;
            else
            {
                return PalindromicNumber(no+rev, itr+1);
            }

            
        }

        public static int BinaryToDecimal(string binary)
        {
            int no=0;
            for (int i = binary.Length-1; i >=0; i--)
            {
                int b = binary[i];


                if(b=='1')
                no = no + (int) Math.Pow(2, binary.Length - 1 - i);
            }

            return no;
        }

        public static void CheckForFever()
        {
            Random rdm = new Random();

            List<float> temp = new List<float>();
            SortedSet<float> temps = new SortedSet<float>();


            for (int i = 0; i < 100; i++)
            {
                var rdmTemp = (float) rdm.NextDouble() * (105 - 94) + 94;
                temp.Add(rdmTemp);

                if (temps.Count < 10)
                    temps.Add(rdmTemp);
                else
                {
                    var lowValue = temps.ElementAt(0);

                    if(rdmTemp>lowValue)
                    {
                        temp.Remove(lowValue);
                        temp.Add(rdmTemp);
                    }
                }
            }

            Console.WriteLine(temps.ToString());
        }

        static void Main(string[] args)
        {
            string text = "Hello how are naresh you";

            string t = "Welcome to paradise";

            var rs = t.Substring(11, 3);

            CheckForFever();
            //Samp();
            //Check();
            //string result = string.Empty;
            StringBuilder result = new StringBuilder();

            //var rslt = SetFour(new int[] { -1, 1, 4, 2, -4, 5, -2 });

            //var rslt = MinCoins(new int[] { 1, 3, 5 }, 9);

            //var rslt = PalindromicNumber(122, 1);

            //var rslt = BinaryToDecimal("101010101");

            //Console.WriteLine(rslt);

            //bool flag = false;

            //for (int i = text.Length-1; i>=0; i--)
            //{
            //    if (text[i] == ' ')
            //    {
            //        if (!flag)
            //        {
            //            flag = true;
            //        }
            //        else
            //            break;
            //    }else if(flag)
            //    {
            //        result = text[i] + result; ;
            //    }

            //    //result = result+ text[i];
            //}
            //char[] vowels = { 'a', 'e', 'i', 'o', 'u' };
            //for (int i = 0; i < text.Length; i++)
            //{
            //    if(text[i]=='a'|| text[i] == 'e' || text[i] == 'i' || text[i] == 'o' || text[i] == 'u')
            //    {
            //        continue;
            //    }
            //    else
            //    {
            //        result.Append(text[i]);
            //    }
            //}

            //string point1 = "(1,1)";
            //string point2 = "(1,3)";

            //int x1, y1, x2, y2;

            //point1 = point1.Replace("(", "").Replace(")", "");
            //point2 = point2.Replace("(", "").Replace(")", "");
            //var pnts1 = point1.Split(',');
            //var pnts2 = point2.Split(',');

            //x1 = int.Parse(pnts1[0]+"");
            //y1 = int.Parse(pnts1[1] + "");
            //x2 = int.Parse(pnts2[0] + "");
            //y2 = int.Parse(pnts2[1] + "");
            

            //if(x1==x2 & y1==y2)
            //{
            //    result.Append("C");
            //}
            //else 
            //if(y1==y2)
            //{
            //    if (x1 < x2)
            //        result.Append("W");
            //    else
            //    {
            //        result.Append("E");
            //    }
            //}else if(x1==x2)
            //{
            //    if (y1 < y2)
            //        result.Append("N");
            //    else
            //    {
            //        result.Append("S");
            //    }
            //}else if(x2>x1 && y2>y1)
            //{
            //    result.Append("NW");
            //}
            //else if (x2 > x1 && y2<y1)
            //{
            //    result.Append("SW");
            //}
            //else if (x2<x1)
            //{
            //    if (y2 > y1)
            //        result.Append("NE");
            //    else
            //        result.Append("SE");
            //}


            Console.WriteLine(result);
            Console.ReadLine();
        }
    }
}
