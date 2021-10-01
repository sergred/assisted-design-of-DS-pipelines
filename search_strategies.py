def bfs(self, pool_of_candidates, config, user_query):
    config['global_limit'] = config.get('num_candidates') + config.get('num_candidates') ** config.get('depth_limit')
    suggestions = self.engine.suggest(pool_of_candidates, config, discarded=config.get('failed_exec', []))
    res, suggested, visited = suggestions[:], defaultdict(int), []
    for c in suggestions:
        suggested[c.fid] += 1

    while suggestions:
        chosen = suggestions.pop(0)
        visited.append(chosen.fid)
        if len(res) >= config.get('global_limit'):
            return res
        new_suggestions = self.engine.suggest(pool_of_candidates, config, offset=chosen.fid, previous=suggested, discarded=config.get('failed_exec', []) + visited)
        for c in new_suggestions:
            suggested[c.fid] += 1

        res.extend(new_suggestions)
        suggestions.extend(new_suggestions)
    return res

def dfs(self, pool_of_candidates, config, user_query):
    suggestions = self.engine.suggest(pool_of_candidates, config)
    res, suggested, visited = [], defaultdict(int), []
    for c in suggestions:
        suggested[c.fid] += 1

    while suggestions:
        chosen = suggestions.pop(0)
        visited.append(chosen.fid)
        res.append(chosen)
        if len(visited) % config.get('depth_limit') == 0:
            continue
        if len(res) > config.get('global_limit'):
            return res
        new_suggestions = self.engine.suggest(pool_of_candidates, config, offset=chosen.fid, previous=suggested, discarded=visited)
        for c in new_suggestions:
            suggested[c.fid] += 1

        suggestions.insert(0, new_suggestions[0])
    return res


def astar(self, pool_of_candidates, config, user_query):
    suggestions = self.engine.suggest(pool_of_candidates, config, discarded=config.get('failed_exec', []))
    res, suggested, visited = suggestions[:], defaultdict(int), []

    for c in suggestions:
        suggested[c.fid] += 1

    while suggestions:
        r = self.eval_suggestions(user_query, suggestions)
        print(r)
        idx = np.nanargmax([v[1] for v in r])
        print("Chosen", suggestions[idx].fid)
        chosen = suggestions.pop(idx)
        print(chosen.fid)
        visited.append(chosen.fid)
        if len(res) > config.get('global_limit'):
            return res

        suggestions = self.engine.suggest(pool_of_candidates, config, offset=chosen.fid, previous=suggested, discarded=config.get('failed_exec', []) + visited)
        for c in suggestions:
            suggested[c.fid] += 1

        res.extend(suggestions)
    return res

def rands(self, pool_of_candidates, config, user_query):
    suggestions = self.engine.suggest(pool_of_candidates, config)
    visited, suggested, res = [], defaultdict(int), []
    while suggestions:
        chosen = suggestions.pop(np.random.randint(len(suggestions)))
        visited.append(chosen.fid)
        res.append(chosen)
        if len(res) > config.get('global_limit'):
            return res
        suggestions = self.engine.suggest(pool_of_candidates, config, offset=chosen.fid, previous=suggested, discarded=visited)
        for c in suggestions:
            suggested[c.fid] += 1
