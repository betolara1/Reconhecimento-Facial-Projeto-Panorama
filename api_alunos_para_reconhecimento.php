<?php
header("Content-Type: application/json; charset=UTF-8");
header("Access-Control-Allow-Origin: *"); // Em produção, restrinja para o domínio da sua aplicação Python

require_once 'conexao.php'; // Inclui a conexão do banco de dados

try {
    // Garante que a conexão use UTF-8
    $conn->exec("SET NAMES 'utf8'");

    // A consulta seleciona alunos ativos que não tenham um CPF vazio
    $sql = "SELECT id, nome_aluno, cpf_aluno FROM alunos WHERE situacao = 'Ativo' AND cpf_aluno IS NOT NULL AND cpf_aluno != ''";
    $stmt = $conn->prepare($sql);
    $stmt->execute();

    // Define o modo de busca para associativo
    $stmt->setFetchMode(PDO::FETCH_ASSOC);
    $alunos = $stmt->fetchAll();

    echo json_encode($alunos);

} catch(PDOException $e) {
    http_response_code(500);
    echo json_encode(["error" => "Erro na consulta SQL: " . $e->getMessage()]);
}

// A conexão PDO é fechada automaticamente no final do script, mas podemos anular o objeto se quisermos.
$conn = null;
?> 