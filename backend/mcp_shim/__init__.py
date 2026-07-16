"""Stdio → Streamable HTTP bridge for the Diarix MCP server.

Some MCP clients only know how to spawn a subprocess and talk to it over
stdin/stdout (the "stdio" transport). This package is a ~150-line adapter:
the client spawns us as ``voicebox-mcp``; we proxy every JSON-RPC frame
to http://127.0.0.1:17494/mcp/ and stream responses back out.

All the real work (tools, models, inference) lives in the Diarix server
process — this package contains no business logic.
"""
