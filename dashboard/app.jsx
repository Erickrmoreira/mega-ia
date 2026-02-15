function App() {
  const [games, setGames] = React.useState([]);
  const [amount, setAmount] = React.useState("");
  const [modality, setModality] = React.useState("");
  const [loading, setLoading] = React.useState(false);
  const [limitWarning, setLimitWarning] = React.useState("");
  const [apiError, setApiError] = React.useState("");

  // Converte string de dezenas em array numerico validado para renderizacao.
  const parseGame = (formatted) => {
    if (typeof formatted !== "string") return [];
    return formatted.split("-").map((x) => parseInt(x.trim(), 10)).filter((n) => !Number.isNaN(n));
  };

  const generateGames = async () => {
    if (!modality || !amount) return;

    setLoading(true);
    setGames([]);
    setLimitWarning("");
    setApiError("");

    try {
      const quantity = parseInt(amount, 10);
      if (!Number.isInteger(quantity) || quantity < 1) {
        setApiError("Informe uma quantidade válida (mínimo 1).");
        return;
      }
      // Frontend reforca o limite contratual de 100 jogos por geracao.
      const effectiveQuantity = Math.min(quantity, 100);
      if (quantity > 100) {
        setLimitWarning("O máximo permitido por geração é 100 jogos. A quantidade foi ajustada automaticamente.");
      }

      const res = await axios.get(`/generate/${modality}`, {
        params: { n_games: effectiveQuantity, candidates: 5000 },
      });

      const returnedGames = (res.data.games || []).map((row) => ({
        numbers: parseGame(row.game),
        predictedScore: row.predicted_score,
      }));
      setGames(returnedGames);
    } catch (e) {
      console.error("Erro detalhado:", e.response?.data);
      // Prioriza mensagem tecnica retornada pela API quando disponível.
      const detail = e.response?.data?.detail;
      setApiError(
        typeof detail === "string" && detail.trim()
          ? detail
          : "Erro na API. Verifique o console do navegador."
      );
    } finally {
      setLoading(false);
    }
  };

  const handleModalityChange = (newModality) => {
    setModality(newModality);
    setGames([]);
    setLimitWarning("");
    setApiError("");
  };

  return (
    <div className="container">
      <header className="header">
        <img src="/images/mega-ia-logo.png" alt="Mega IA" className="logo" />
      </header>

      <div className="controls">
        <div className="control-group">
          <label>Modalidade</label>
          <select
            value={modality}
            onChange={(e) => handleModalityChange(e.target.value)}
            className="modality-select"
            style={{ paddingRight: "52px", color: modality ? "#ffffff" : "#64748b" }}
          >
            <option value="" disabled>
              Selecione o tipo de jogo
            </option>
            <option value="mega-sena">Mega-Sena</option>
            <option value="quina">Quina</option>
            <option value="lotofacil">Lotofacil</option>
          </select>
        </div>

        <div className="control-group">
          <label>Quantidade</label>
          <input
            type="number"
            min="1"
            max="9999"
            value={amount}
            onChange={(e) => {
              setAmount(e.target.value);
              setApiError("");
            }}
            placeholder="Ex: 5"
          />
        </div>

        <div className="control-group button-group">
          <label>&nbsp;</label>
          <button onClick={generateGames} disabled={loading || !modality || !amount}>
            {loading ? "Gerando..." : "Gerar Jogos"}
          </button>
        </div>
      </div>

      {limitWarning && (
        <div style={{ margin: "12px auto 24px", textAlign: "center", color: "#facc15", fontSize: "0.95rem", fontWeight: "500" }}>
          {limitWarning}
        </div>
      )}

      {apiError && (
        <div style={{ margin: "8px auto 24px", textAlign: "center", color: "#f87171", fontSize: "0.95rem", fontWeight: "500" }}>
          {apiError}
        </div>
      )}

      <div className="games">
        {games.length > 0 &&
          games.map((game, i) => (
            <div key={i} className="game" data-modality={modality}>
              <small className="game-label">
                Jogo {String(i + 1).padStart(2, "0")} | score: {Number(game.predictedScore || 0).toFixed(3)}
              </small>
              <div className="balls-container" style={{ gap: modality === "lotofacil" ? "6px" : "8px", justifyContent: "center" }}>
                {game.numbers.map((n, idx) => (
                  <span key={idx} className={`number-ball ${modality}-ball`}>
                    {String(n).padStart(2, "0")}
                  </span>
                ))}
              </div>
            </div>
          ))}
      </div>
    </div>
  );
}

const rootElement = document.getElementById("root");
if (rootElement) {
  const root = ReactDOM.createRoot(rootElement);
  root.render(<App />);
}
