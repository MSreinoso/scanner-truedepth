import Foundation
import Combine
import Network
import Darwin

/// Network state belongs exclusively to streamQueue; SwiftUI state to main.
final class BinaryStreamClient: ObservableObject {
    @Published var connectionState = "Desconectado"
    @Published var isConnected = false
    @Published var framesStreamed = 0
    @Published var serverIP = ""

    private let streamQueue = DispatchQueue(label: "com.bichito3d.stream")
    private var connection: NWConnection?
    private var udpListener: NWListener?
    private var discoveryConnections: [UUID: NWConnection] = [:]
    private var reconnectTask: DispatchWorkItem?
    private var targetIP = ""
    private var ready = false
    private var sending = false
    private var stopped = false
    private var discoveryStarted = false

    private func publish(_ state: String, connected: Bool = false) {
        DispatchQueue.main.async { [weak self] in
            self?.connectionState = state
            self?.isConnected = connected
        }
    }

    func startDiscovery() {
        streamQueue.async { [weak self] in
            guard let self = self, !self.discoveryStarted else { return }
            self.discoveryStarted = true
            self.stopped = false
            self.startUDPListener()
        }
    }

    private func startUDPListener() {
        guard !stopped else { return }
        do {
            let params = NWParameters.udp
            params.allowLocalEndpointReuse = true
            let listener = try NWListener(using: params, on: 50505)
            udpListener = listener
            listener.stateUpdateHandler = { [weak self, weak listener] state in
                guard let self = self, let listener = listener,
                      self.udpListener === listener, !self.stopped else { return }
                if case .failed = state {
                    self.udpListener = nil
                    listener.cancel()
                    self.streamQueue.asyncAfter(deadline: .now() + 3) { [weak self] in
                        self?.startUDPListener()
                    }
                }
            }
            listener.newConnectionHandler = { [weak self] conn in
                guard let self = self, !self.stopped else { conn.cancel(); return }
                let id = UUID()
                // Keep UDP peers alive: later datagrams reuse the same flow.
                guard self.discoveryConnections.count < 16 else { conn.cancel(); return }
                self.discoveryConnections[id] = conn
                conn.start(queue: self.streamQueue)
                self.receiveBeacon(conn, id: id)
            }
            listener.start(queue: streamQueue)
        } catch {
            publish("Descubrimiento no disponible. Usa conexión manual.")
        }
    }

    private func receiveBeacon(_ conn: NWConnection, id: UUID) {
        conn.receiveMessage { [weak self, weak conn] data, _, _, error in
            guard let self = self, let conn = conn else { return }
            guard error == nil, !self.stopped else {
                self.discoveryConnections.removeValue(forKey: id)
                conn.cancel()
                return
            }
            if let data = data, let message = String(data: data, encoding: .utf8),
               message.hasPrefix("BICHITO3D_SERVER|"),
               case let .hostPort(host, _) = conn.endpoint,
               self.connection == nil, self.reconnectTask == nil {
                // Use the datagram's actual sender, avoiding advertised VPN IPs.
                self.connectOnQueue(to: "\(host)")
            }
            self.receiveBeacon(conn, id: id)
        }
    }

    func connect(to ip: String) {
        let trimmed = ip.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return }
        streamQueue.async { [weak self] in self?.connectOnQueue(to: trimmed) }
    }

    private func connectOnQueue(to ip: String) {
        if targetIP == ip, connection != nil { return }
        stopped = false
        reconnectTask?.cancel()
        reconnectTask = nil
        let old = connection
        connection = nil // Ignore callbacks from the old connection.
        old?.cancel()
        ready = false
        sending = false
        targetIP = ip
        DispatchQueue.main.async { [weak self] in self?.serverIP = ip }
        publish("Conectando a \(ip):50506…")
        let tcp = NWProtocolTCP.Options()
        tcp.noDelay = true
        tcp.enableKeepalive = true
        tcp.keepaliveIdle = 5
        let conn = NWConnection(host: NWEndpoint.Host(ip), port: 50506,
                                using: NWParameters(tls: nil, tcp: tcp))
        connection = conn
        conn.stateUpdateHandler = { [weak self, weak conn] state in
            guard let self = self, let conn = conn, self.connection === conn else { return }
            switch state {
            case .ready:
                self.ready = true
                self.publish("Conectado", connected: true)
                self.watchForClose(conn)
            case .waiting(let error), .failed(let error):
                self.fail(conn, message: "Sin conexión: \(error.localizedDescription)")
            case .cancelled:
                self.fail(conn, message: "Desconectado")
            default: break
            }
        }
        conn.start(queue: streamQueue)
        streamQueue.asyncAfter(deadline: .now() + 8) { [weak self, weak conn] in
            guard let self = self, let conn = conn, self.connection === conn, !self.ready else { return }
            self.fail(conn, message: "Tiempo agotado. Revisa IP, red local y firewall.")
        }
    }

    private func watchForClose(_ conn: NWConnection) {
        conn.receive(minimumIncompleteLength: 1, maximumLength: 1024) { [weak self, weak conn] _, _, complete, error in
            guard let self = self, let conn = conn, self.connection === conn else { return }
            if complete || error != nil {
                self.fail(conn, message: "PC desconectado; reintentando…")
            } else {
                self.watchForClose(conn)
            }
        }
    }

    private func fail(_ conn: NWConnection, message: String) {
        guard connection === conn else { return }
        connection = nil
        ready = false
        sending = false
        conn.cancel()
        publish(message)
        guard !stopped else { return }
        let ip = targetIP
        let task = DispatchWorkItem { [weak self] in
            guard let self = self, !self.stopped else { return }
            self.reconnectTask = nil
            self.connectOnQueue(to: ip)
        }
        reconnectTask = task
        streamQueue.asyncAfter(deadline: .now() + 2, execute: task)
    }

    /// The camera calls this synchronously on its processing queue. Never queue
    /// packets behind a slow send: at most ONE packet may be in flight.
    var canAcceptFrame: Bool {
        streamQueue.sync { ready && !sending }
    }

    func sendFrame(_ data: Data) {
        streamQueue.sync {
            guard ready, !sending, let conn = connection else { return }
            sending = true
            conn.send(content: data, completion: .contentProcessed { [weak self, weak conn] error in
                guard let self = self, let conn = conn, self.connection === conn else { return }
                self.sending = false
                if let error = error {
                    self.fail(conn, message: "Error de envío: \(error.localizedDescription)")
                } else {
                    DispatchQueue.main.async { [weak self] in self?.framesStreamed += 1 }
                }
            })
        }
    }

    func disconnect() {
        streamQueue.async { [weak self] in
            guard let self = self else { return }
            self.stopped = true
            self.discoveryStarted = false
            self.reconnectTask?.cancel()
            self.reconnectTask = nil
            let old = self.connection
            self.connection = nil
            old?.cancel()
            self.udpListener?.cancel()
            self.udpListener = nil
            self.discoveryConnections.values.forEach { $0.cancel() }
            self.discoveryConnections.removeAll()
            self.ready = false
            self.sending = false
            self.publish("Desconectado")
        }
    }

    deinit {
        reconnectTask?.cancel()
        connection?.cancel()
        udpListener?.cancel()
        discoveryConnections.values.forEach { $0.cancel() }
    }
}
