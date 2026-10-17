import Foundation
import Network

// MARK: - BinaryStreamClient
/// A persistent TCP binary streaming client for the 3D scanner.
///
/// This class maintains a long-lived TCP connection to a PC server and streams
/// serialized frame packets over it. It also handles UDP-based discovery:
///   - Listens for server beacons on UDP port 50505 to auto-connect.
///   - Broadcasts its own beacon so the PC studio can discover the phone.
///
/// All `NWConnection` / `NWListener` operations run on a dedicated serial queue
/// (`streamQueue`). All `@Published` property updates are dispatched to the main
/// queue to keep SwiftUI happy.
class BinaryStreamClient: ObservableObject {

    // MARK: - Published Properties

    /// Human-readable connection state string for the UI.
    @Published var connectionState: String = "Desconectado"

    /// Whether a TCP connection to the server is currently established.
    @Published var isConnected: Bool = false

    /// Running count of frames successfully queued for sending.
    @Published var framesStreamed: Int = 0

    /// The IP address of the connected (or last-known) PC server.
    @Published var serverIP: String = ""

    // MARK: - Private Properties

    /// The active TCP connection to the PC server.
    private var connection: NWConnection?

    /// Dedicated serial queue for all network I/O to guarantee thread safety.
    private let streamQueue = DispatchQueue(label: "com.bichito3d.streamQueue", qos: .userInteractive)

    /// TCP port that the PC server listens on for binary frame data.
    private let serverPort: UInt16 = 50506

    /// UDP listener used for server discovery beacons.
    private var udpListener: NWListener?

    /// UDP port used for discovery (both listening and broadcasting).
    private let udpDiscoveryPort: UInt16 = 50505

    /// Timer that drives the periodic UDP beacon broadcast.
    private var beaconTimer: DispatchSourceTimer?

    /// Flag to prevent multiple concurrent reconnection attempts.
    private var isReconnecting: Bool = false

    /// Flag to indicate the client was intentionally disconnected.
    private var intentionalDisconnect: Bool = false

    /// UDP connection used for broadcasting beacons to the PC.
    private var udpBroadcastConnection: NWConnection?

    // MARK: - Discovery

    /// Start the UDP discovery system.
    ///
    /// This does two things:
    /// 1. Listens on UDP port 50505 for server beacons of the form
    ///    `BICHITO3D_SERVER|IP:x.x.x.x|PORT:50506`. When one is received the
    ///    client automatically connects to that server.
    /// 2. Starts broadcasting a phone-side beacon every 1.5 seconds so the PC
    ///    studio can also discover this device.
    func startDiscovery() {
        startUDPListener()
        startUDPBeacon()
    }

    /// Listen for `BICHITO3D_SERVER` beacons on UDP port 50505.
    private func startUDPListener() {
        // Tear down any existing listener first.
        udpListener?.cancel()
        udpListener = nil

        let params = NWParameters.udp
        params.allowLocalEndpointReuse = true
        // Enable port reuse so the beacon broadcaster can share the port.
        params.requiredLocalEndpoint = NWEndpoint.hostPort(host: .ipv4(.any), port: NWEndpoint.Port(rawValue: udpDiscoveryPort)!)

        do {
            let listener = try NWListener(using: params)

            listener.stateUpdateHandler = { [weak self] state in
                switch state {
                case .ready:
                    print("[BinaryStreamClient] UDP discovery listener ready on port \(self?.udpDiscoveryPort ?? 0)")
                case .failed(let error):
                    print("[BinaryStreamClient] UDP listener failed: \(error). Restarting...")
                    self?.udpListener?.cancel()
                    // Retry after a short delay.
                    self?.streamQueue.asyncAfter(deadline: .now() + 2.0) {
                        self?.startUDPListener()
                    }
                case .cancelled:
                    print("[BinaryStreamClient] UDP listener cancelled")
                default:
                    break
                }
            }

            listener.newConnectionHandler = { [weak self] newConnection in
                self?.handleDiscoveryConnection(newConnection)
            }

            listener.start(queue: streamQueue)
            udpListener = listener

        } catch {
            print("[BinaryStreamClient] Failed to create UDP listener: \(error)")
        }
    }

    /// Handle an incoming UDP datagram on the discovery port.
    ///
    /// Expected beacon format: `BICHITO3D_SERVER|IP:192.168.x.x|PORT:50506`
    private func handleDiscoveryConnection(_ conn: NWConnection) {
        conn.start(queue: streamQueue)
        conn.receiveMessage { [weak self] data, _, _, error in
            defer { conn.cancel() }

            guard let self = self, let data = data,
                  let message = String(data: data, encoding: .utf8) else {
                return
            }

            print("[BinaryStreamClient] UDP beacon received: \(message)")

            // Parse the server beacon.
            guard message.hasPrefix("BICHITO3D_SERVER") else { return }

            let parts = message.components(separatedBy: "|")
            var discoveredIP: String?

            for part in parts {
                if part.hasPrefix("IP:") {
                    discoveredIP = String(part.dropFirst(3))
                }
            }

            guard let ip = discoveredIP else {
                print("[BinaryStreamClient] Could not parse IP from beacon")
                return
            }

            // Only connect if we aren't already connected to this server.
            if !self.isConnected || self.serverIP != ip {
                print("[BinaryStreamClient] Auto-connecting to discovered server at \(ip)")
                self.connect(to: ip)
            }
        }
    }

    // MARK: - TCP Connection

    /// Create a persistent TCP connection to the PC server.
    ///
    /// The connection is set up with state-change handlers that automatically
    /// update the published UI properties and trigger reconnection on failure.
    ///
    /// - Parameter ip: The IPv4 address of the PC server.
    func connect(to ip: String) {
        intentionalDisconnect = false

        // Tear down any existing connection.
        connection?.cancel()
        connection = nil

        DispatchQueue.main.async {
            self.serverIP = ip
            self.connectionState = "Conectando..."
        }

        let host = NWEndpoint.Host(ip)
        let port = NWEndpoint.Port(rawValue: serverPort)!

        // Use TCP with no special options.
        let tcp = NWProtocolTCP.Options()
        tcp.noDelay = true                  // Disable Nagle's algorithm for low latency.
        tcp.enableKeepalive = true
        tcp.keepaliveIdle = 5               // Start keepalive probes after 5 s idle.

        let params = NWParameters(tls: nil, tcp: tcp)

        let conn = NWConnection(host: host, port: port, using: params)

        conn.stateUpdateHandler = { [weak self] state in
            guard let self = self else { return }

            switch state {
            case .ready:
                print("[BinaryStreamClient] TCP connection ready to \(ip):\(self.serverPort)")
                self.isReconnecting = false
                DispatchQueue.main.async {
                    self.isConnected = true
                    self.connectionState = "Conectado"
                }

            case .waiting(let error):
                print("[BinaryStreamClient] TCP connection waiting: \(error)")
                DispatchQueue.main.async {
                    self.connectionState = "Esperando..."
                }

            case .failed(let error):
                print("[BinaryStreamClient] TCP connection failed: \(error)")
                DispatchQueue.main.async {
                    self.isConnected = false
                    self.connectionState = "Desconectado"
                }
                self.scheduleReconnect(to: ip)

            case .cancelled:
                print("[BinaryStreamClient] TCP connection cancelled")
                DispatchQueue.main.async {
                    self.isConnected = false
                    self.connectionState = "Desconectado"
                }
                if !self.intentionalDisconnect {
                    self.scheduleReconnect(to: ip)
                }

            default:
                break
            }
        }

        conn.start(queue: streamQueue)
        connection = conn
    }

    /// Schedule a reconnection attempt after a 2-second delay.
    ///
    /// Multiple calls while a reconnect is already pending are no-ops.
    private func scheduleReconnect(to ip: String) {
        guard !isReconnecting, !intentionalDisconnect else { return }
        isReconnecting = true

        print("[BinaryStreamClient] Scheduling reconnect to \(ip) in 2 seconds...")

        streamQueue.asyncAfter(deadline: .now() + 2.0) { [weak self] in
            guard let self = self, !self.intentionalDisconnect else { return }
            self.isReconnecting = false
            self.connect(to: ip)
        }
    }

    // MARK: - Frame Streaming

    /// Send a serialized binary frame over the TCP connection.
    ///
    /// This call is **non-blocking**. The data is handed off to Network.framework's
    /// internal send buffer. If an error occurs during sending the connection is
    /// automatically torn down and a reconnect is scheduled.
    ///
    /// - Parameter data: The fully serialized binary frame packet (header + payload).
    func sendFrame(_ data: Data) {
        guard let conn = connection, isConnected else { return }

        conn.send(content: data, completion: .contentProcessed { [weak self] error in
            if let error = error {
                print("[BinaryStreamClient] Send error: \(error)")
                guard let self = self else { return }
                let ip = self.serverIP
                DispatchQueue.main.async {
                    self.isConnected = false
                    self.connectionState = "Error de envío"
                }
                self.connection?.cancel()
                self.connection = nil
                self.scheduleReconnect(to: ip)
            } else {
                // Successfully queued for sending — bump the counter.
                DispatchQueue.main.async {
                    self?.framesStreamed += 1
                }
            }
        })
    }

    // MARK: - Disconnect

    /// Gracefully disconnect from the server and stop all discovery activity.
    func disconnect() {
        intentionalDisconnect = true
        isReconnecting = false

        // Stop the beacon timer.
        beaconTimer?.cancel()
        beaconTimer = nil

        // Stop the broadcast connection.
        udpBroadcastConnection?.cancel()
        udpBroadcastConnection = nil

        // Stop the UDP listener.
        udpListener?.cancel()
        udpListener = nil

        // Tear down the TCP connection.
        connection?.cancel()
        connection = nil

        DispatchQueue.main.async {
            self.isConnected = false
            self.connectionState = "Desconectado"
            self.framesStreamed = 0
        }

        print("[BinaryStreamClient] Disconnected and stopped discovery")
    }

    // MARK: - UDP Beacon Broadcasting

    /// Broadcast a phone-side discovery beacon every 1.5 seconds.
    ///
    /// The beacon is sent as a UDP broadcast on port 50505 with the format:
    /// `TRUEDEPTH_3D_BEACON|IP:{localIP}|PORT:8080|STREAMER_V2`
    ///
    /// This allows the PC studio to discover the phone on the local network
    /// (backward-compatible with the older HTTP-based streaming path).
    func startUDPBeacon() {
        // Tear down any previous beacon timer.
        beaconTimer?.cancel()
        beaconTimer = nil
        udpBroadcastConnection?.cancel()
        udpBroadcastConnection = nil

        // Create a UDP connection targeting the broadcast address.
        let params = NWParameters.udp
        params.allowLocalEndpointReuse = true

        // Enable broadcast by setting the IP-level socket option.
        if let ipOptions = params.defaultProtocolStack.internetProtocol as? NWProtocolIP.Options {
            // Network.framework doesn't expose SO_BROADCAST directly, but
            // sending to 255.255.255.255 works on iOS for UDP.
        }

        let broadcastConn = NWConnection(
            host: NWEndpoint.Host("255.255.255.255"),
            port: NWEndpoint.Port(rawValue: udpDiscoveryPort)!,
            using: params
        )

        broadcastConn.stateUpdateHandler = { state in
            switch state {
            case .ready:
                print("[BinaryStreamClient] UDP beacon broadcast connection ready")
            case .failed(let error):
                print("[BinaryStreamClient] UDP beacon broadcast failed: \(error)")
            default:
                break
            }
        }

        broadcastConn.start(queue: streamQueue)
        udpBroadcastConnection = broadcastConn

        // Set up a repeating timer on the stream queue.
        let timer = DispatchSource.makeTimerSource(queue: streamQueue)
        timer.schedule(deadline: .now() + 1.0, repeating: 1.5)

        timer.setEventHandler { [weak self] in
            guard let self = self else { return }

            guard let localIP = self.getWiFiAddress() else {
                print("[BinaryStreamClient] Cannot broadcast beacon — no Wi-Fi address")
                return
            }

            let beacon = "TRUEDEPTH_3D_BEACON|IP:\(localIP)|PORT:8080|STREAMER_V2"
            guard let data = beacon.data(using: .utf8) else { return }

            self.udpBroadcastConnection?.send(content: data, completion: .contentProcessed { error in
                if let error = error {
                    print("[BinaryStreamClient] Beacon send error: \(error)")
                }
            })
        }

        timer.resume()
        beaconTimer = timer
    }

    // MARK: - Network Utilities

    /// Retrieve the device's Wi-Fi IPv4 address.
    ///
    /// Checks `en0` (Wi-Fi) and `bridge100` (Personal Hotspot / USB tethering)
    /// interfaces and returns the first valid IPv4 address found.
    ///
    /// - Returns: A string containing the IPv4 address, or `nil` if unavailable.
    private func getWiFiAddress() -> String? {
        var address: String?
        var ifaddr: UnsafeMutablePointer<ifaddrs>?

        guard getifaddrs(&ifaddr) == 0, let firstAddr = ifaddr else {
            return nil
        }

        defer { freeifaddrs(ifaddr) }

        for ptr in sequence(first: firstAddr, next: { $0.pointee.ifa_next }) {
            let interface = ptr.pointee
            let addrFamily = interface.ifa_addr.pointee.sa_family

            // Only interested in IPv4 (AF_INET).
            guard addrFamily == UInt8(AF_INET) else { continue }

            let name = String(cString: interface.ifa_name)
            guard name == "en0" || name == "bridge100" else { continue }

            // Convert the sockaddr to a human-readable string.
            var hostname = [CChar](repeating: 0, count: Int(NI_MAXHOST))
            if getnameinfo(
                interface.ifa_addr,
                socklen_t(interface.ifa_addr.pointee.sa_len),
                &hostname,
                socklen_t(hostname.count),
                nil, 0,
                NI_NUMERICHOST
            ) == 0 {
                address = String(cString: hostname)
                // Prefer en0 over bridge100 — break immediately if found.
                if name == "en0" { break }
            }
        }

        return address
    }

    // MARK: - Cleanup

    deinit {
        disconnect()
    }
}
