using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Net;
using System.Net.Sockets;
using System.IO;
using System.Text.RegularExpressions;

namespace MailScrap
{
    class MailScrap
    {
        private TcpClient _imapClient;
        private NetworkStream _imapNs;
        private StreamWriter _imapSw;
        private StreamReader _imapSr;

        public void InitializeConnection(string hostname, int port)
        {
            try
            {
                _imapClient = new TcpClient(hostname, port);
                _imapNs = _imapClient.GetStream();
                _imapSw = new StreamWriter(_imapNs);
                _imapSr = new StreamReader(_imapNs);

                Console.WriteLine("*** Connected ***");
                  Response();
            }
            catch (SocketException ex)
            {
                Console.WriteLine(ex.Message);
                //return ex.Message;
            }
        }
        public string Response()
        {
            byte[] data = new byte[_imapClient.ReceiveBufferSize];
            int ret = _imapNs.Read(data, 0, data.Length);
            return Encoding.ASCII.GetString(data).TrimEnd();
        }

        public string AuthenticateUser(string username, string password)
        {
            _imapSw.WriteLine("$ LOGIN " + username + " " + password);
            _imapSw.Flush();
            return Response();
        }
    }
}
