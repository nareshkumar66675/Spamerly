using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using AE.Net.Mail;

namespace MailScrap
{
    class Program
    {

        static void Main(string[] args)
        {
            //MailScrap mailScrap = new MailScrap();

            //mailScrap.InitializeConnection("imap.gmail.com", 993);

            //mailScrap.AuthenticateUser("nareshkumar66675@gmail.com", "naresh14GMAIL");

            Console.WriteLine("Enter your Gmail Address:");
            var addr = Console.ReadLine();

            Console.WriteLine("Enter your Gmail Password(Trust me it is safe):");
            var pwd = Console.ReadLine();

            Console.WriteLine("Folder Path");
            var path = Console.ReadLine();

            using (var imap = new AE.Net.Mail.ImapClient("imap.gmail.com", addr, pwd, AuthMethods.Login, 993,true))
            {
                //var mb = imap.ListMailboxes("", "*");
                imap.SelectMailbox("[Gmail]/Spam");


                var count = imap.GetMessageCount();

                Console.WriteLine($"Spam Count: {count}");

                var msgs = imap.GetMessages(0, count - 1, false);
                int i = 0;
                foreach (var msg in msgs)
                {
                    File.WriteAllText($@"{path}\Spam{i}.txt", msg.Body);
                    i++;
                }

                //var msgs = imap.SearchMessages(
                //  SearchCondition.Undeleted().And(
                //    SearchCondition.From("david"),
                //    SearchCondition.SentSince(new DateTime(2000, 1, 1))
                //  ).Or(SearchCondition.To("andy"))
                //);

            //Flags.
            }
            Console.WriteLine("You have been Hacked ;P");
            Console.WriteLine("Press any key.");
            Console.ReadLine();
        }
    }
}
